import optax
import jax
import flax
import jax.numpy as jnp

from typing import Any, Callable, Optional, Union, NamedTuple
from optax._src.base import Params, identity
from optax._src import combine
from optax._src.alias import _scale_by_learning_rate, ScalarOrSchedule
from optax._src import transform
from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax._src.transform import update_moment, bias_correction, ScaleByAdamState, update_moment_per_elem_norm

from flax.training.train_state import TrainState

from jax_utils import copy_pytree, compute_mvn_kl, weighted_kl, compute_global_norm



def training_losses_fn(params, rng, state, train_inputs, global_sr_weight, sigma_q, rate_schedule): 
    img, label = train_inputs
    if img.shape[1]==256:
        img = jax.image.resize(img, shape=[img.shape[0], 32, 32, img.shape[-1]], method='bilinear')

    outputs, singular_vectors = state.apply_fn({'params': params}, rng=rng, img=img, label=label, mutable=['singular_vectors'])
    model_out, unweighted_kls, sr_loss = outputs
    sr_loss *= global_sr_weight

    mean_output, var_output = jnp.split(model_out, 2, axis=-1)
    var_output = jnp.exp(var_output)
    qvar = (sigma_q ** 2)
    neg_logpx_z = compute_mvn_kl(img, qvar, mean_output, var_output, raxis=None) #shape is ()

    total_kl_per_image = jnp.sum(jnp.stack(unweighted_kls, axis=-1), axis=-1)  #(B, )
    KL_Loss = jnp.float32(0.)
    for i, k in enumerate(unweighted_kls):
        w = rate_schedule[i]
        k_weighted = weighted_kl(k, total_kl_per_image, w, 2.*w)
        KL_Loss += k_weighted    

    loss_divisor = img.shape[0] 
    neg_logpx_z /= loss_divisor
    KL_Loss /= loss_divisor
    total_loss = neg_logpx_z + KL_Loss + sr_loss

    metrics = {'loss': total_loss, 'distortion term': neg_logpx_z, 'kl term': KL_Loss, 'sr loss': sr_loss}
    return total_loss, (metrics, singular_vectors)

def safe_update(state, grads, vectors, global_norm, clip_value):
    def update(_):
        return state.apply_gradients(grads_and_vectors=(grads,vectors)) 

    def do_nothing(_):
        return state

    state = jax.lax.cond(global_norm < clip_value, update, do_nothing, operand=None)
    skip_bool = jnp.logical_or(global_norm >= clip_value, jnp.isnan(global_norm))  
    return state, jnp.int32(skip_bool)

def train_step_fn(rng, state, train_inputs,  loss, pxz, kl, gnorm, srloss, skip_counter,  training_losses, skip_threshold, n_accums):


    grad_fn = jax.value_and_grad(training_losses, has_aux=True, argnums=0)
    (_, metrics_and_vectors), grads = grad_fn(state.params, rng, state, train_inputs)
    metrics, vectors = metrics_and_vectors

    grads = jax.lax.pmean(grads, axis_name='shards')
    global_norm = compute_global_norm(grads)

    def update_func(_):
        if n_accums > 1:
            return state.update_gradients(grads_and_vectors=(grads, vectors))
        else:
            return state.apply_gradients(grads_and_vectors=(grads, vectors))

    def do_nothing(_):
        return state


    state = jax.lax.cond(
        jnp.logical_or(global_norm < skip_threshold, state.step < 1000), 
        update_func, do_nothing, operand=None
    )

    skip_bool = jnp.logical_or(
        jnp.logical_and(global_norm >= skip_threshold, state.step >= 1000),
        jnp.isnan(global_norm)
    )

    if n_accums > 1:
        state = accum_apply(state, n_accums)

    loss += metrics['loss']
    pxz += metrics['distortion term']
    kl += metrics['kl term']
    srloss += metrics['sr loss']
    gnorm += global_norm
    skip_counter += jnp.int32(skip_bool)
    return state,  loss, pxz, kl, gnorm, srloss, skip_counter

def accum_apply(state, n_accums):
    def update(_):
        return state.apply_gradients() 

    def do_nothing(_):
        return state

    state = jax.lax.cond(state.accum_step>=n_accums, update, do_nothing, operand=None)
    return state



#optimizer related stuff

def update_infinite_moment(updates, moments, decay, eps):
    """Compute the exponential moving average of the infinite moment."""
    return jax.tree_map(
        lambda g, t: jnp.maximum(decay * t, jnp.abs(g) + eps), updates, moments)  # max(β2 · ut−1, |gt|)


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    nu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:

  mu_dtype = utils.canonicalize_dtype(mu_dtype)
  nu_dtype = utils.canonicalize_dtype(nu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map( 
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_util.tree_map(  
        lambda t: jnp.zeros_like(t, dtype=nu_dtype), params)
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)


  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state.mu, b1, 1)
    nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)
    updates = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    mu = utils.cast_tree(mu, mu_dtype)
    nu = utils.cast_tree(nu, nu_dtype)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)



def adam(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,    
    nu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:

  return combine.chain(
      scale_by_adam(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype, nu_dtype=nu_dtype),
      _scale_by_learning_rate(learning_rate),
  )
  

class EMATrainState(TrainState):
    ema_decay: float
    ema_params: Optional[flax.core.FrozenDict[str, Any]]
    singular_vectors: flax.core.FrozenDict[str, Any]

    def update_gradients(self, *, grads_and_vectors, **kwargs):
        raise RuntimeError("The 'update_gradients' method only works for a TrainState object that does gradient accumulation.")

    def apply_gradients(self, *, grads_and_vectors, **kwargs):
        grads, vectors = grads_and_vectors
        vectors = vectors['singular_vectors']

        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        if self.ema_params is not None:
            new_ema_params = jax.tree_map(lambda ema, p: ema * self.ema_decay + (1 - self.ema_decay) * p,
                                      self.ema_params, new_params)
        else:
            new_ema_params = None
        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            singular_vectors=vectors
        )

    @classmethod
    def create(cls, *, apply_fn, params, ema_params, tx, ema_decay, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            ema_params=ema_params,
            tx=tx,
            opt_state=opt_state,
            ema_decay=ema_decay,
            **kwargs,
        )

class EMATrainStateAccum(TrainState):
    ema_decay: float
    ema_params: flax.core.FrozenDict[str, Any]
    singular_vectors: flax.core.FrozenDict[str, Any]
    current_grads: flax.core.FrozenDict[str, Any]
    accum_step: int=0
    
    def update_gradients(self, *, grads_and_vectors, **kwargs):
        grads, vectors = grads_and_vectors
        vectors = vectors['singular_vectors']
        
        new_grads = jax.tree_util.tree_map(lambda x,y: x+y, self.current_grads, grads)
        return self.replace(
            step=self.step,
            params=self.params,
            ema_params=self.ema_params,
            opt_state=self.opt_state,
            singular_vectors=vectors,
            current_grads=new_grads,
            accum_step=self.accum_step+1
        )
        

    def apply_gradients(self):
        updates, new_opt_state = self.tx.update(
            self.current_grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_ema_params = jax.tree_util.tree_map(lambda ema, p: ema * self.ema_decay + (1 - self.ema_decay) * p,
                                      self.ema_params, new_params)
        
        zero_grad = copy_pytree(self.params)
        zero_grad = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), zero_grad)
        
        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            singular_vectors=self.singular_vectors,
            current_grads=zero_grad,
            accum_step=self.accum_step * 0,
        )

    @classmethod
    def create(cls, *, apply_fn, params, ema_params, tx, ema_decay, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            ema_params=ema_params,
            tx=tx,
            opt_state=opt_state,
            ema_decay=ema_decay,
            **kwargs,
        )



class CosineDecay:
    def __init__(self, startlr, maxlr, minlr, warmup_steps, decay_steps):
        self.startlr = startlr
        self.maxlr = maxlr
        self.minlr = minlr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        
    def __call__(self, step):
        step = jnp.minimum(step, self.decay_steps)
        startlr, maxlr, minlr = self.startlr, self.maxlr, self.minlr
        warmup = startlr + step/self.warmup_steps * (maxlr - startlr)

        decay_factor = 0.5 * (1 + jnp.cos(jnp.pi * step/self.decay_steps))
        decay_factor = (1 - minlr/maxlr) * decay_factor + minlr/maxlr
        lr = maxlr * decay_factor
        return jnp.minimum(warmup, lr)

