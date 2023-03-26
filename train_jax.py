from glob import glob
import jax
import jax.numpy as jnp
import numpy as np
import flax
from jax import random
import time
import os
from functools import partial
from tensorflow.config import set_visible_devices as tf_set_visible_devices
from tensorflow.io import gfile, write_file

from absl import app, flags
from ml_collections.config_flags import config_flags

tf_set_visible_devices([], device_type="GPU")
np.set_printoptions(precision=4)
jnp.set_printoptions(precision=4)

from training_utils import CosineDecay, EMATrainState, EMATrainStateAccum, adam, training_losses_fn, train_step_fn
from jax_utils import unreplicate, copy_pytree, count_params, save_checkpoint, restore_checkpoint
from general_utils import get_rate_schedule, print_and_log, denormalize, save_images
from model import VAE, SuperResolutionVAE
from dataset_utils import create_dataset

args = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "the location of the config path you will use to train the model. e.g. ./config/cifar10.py")
flags.DEFINE_string("global_dir", None, "the global directory you will save all training stuff into.")
flags.DEFINE_string("data_dir", None, "the directory where your data is stored (or where it will be downloaded into).")
flags.mark_flags_as_required(["config", "global_dir", "data_dir"])

def generation_step(params, rng, label, img_lr, num, margs, model):
    vae = VAE(**dict(margs))

    outputs, _ = vae.apply({'params': params}, rng=rng, num=num, 
        label=label, img_lr=img_lr,
        method=model.p_sample, mutable=['singular_vectors']
    )
    return outputs


def main(_):
    #setup basic config stuff
    config, global_dir = args.config, args.global_dir
    config.unlock()

    if config.model:
        margs = config.model
    else:
        margs = config.srmodel
    dargs = config.dataset
    targs = config.training
    oargs = config.optimizer

    if not gfile.isdir(global_dir):
        gfile.makedirs(global_dir)
    
    dargs.data_dir = dargs.data_dir.format(args.data_dir)
    targs.checkpoint_dirs = [subdir.format(global_dir) for subdir in targs.checkpoint_dirs]
    targs.log_dir = targs.log_dir.format(global_dir)
    dargs.framework = "JAX"
    n_accums = targs.n_accums

    dataset = create_dataset(dargs)
    #create logfile
    logfile_path = os.path.join(targs.log_dir, 'logfile.txt')
    if not gfile.exists(logfile_path):
        write_file(logfile_path, "")
    printl = partial(print_and_log, logfile_path=logfile_path)

    #create rate schedule
    rate_schedule = get_rate_schedule(targs.rate_schedule, margs.nlayers)
    print("The rate schedule, on a per-layer information level, is: ", rate_schedule)

    total_datadim = dargs.channels * dargs.resolution**2
    skip_threshold = targs.skip_threshold * total_datadim
    training_losses = partial(training_losses_fn, global_sr_weight=targs.global_sr_weight, sigma_q=targs.sigma_q, rate_schedule=rate_schedule)
    train_step = partial(train_step_fn, training_losses=training_losses, skip_threshold=skip_threshold, n_accums=n_accums)

    #set devices, init rng, and spectral regularization vectors
    devices = jax.devices()
    print("Devices:", devices)

    rng = random.PRNGKey(123)
    rng, sample_key, init_key = random.split(rng, num=3)   

    if config.model: model = VAE(**margs)
    else: model = SuperResolutionVAE(**margs)
    print('The model is an instance of', type(model))

    dummy_data, dummy_label = unreplicate(next(dataset)) #(b, h, w, c), (b,)
    print('Data shape', dummy_data.shape)
    dummy_label_init = dummy_label[:2] if dummy_label else None
    if dargs.dataset_name == 'churchbasecascade':
        dummy_data =  jax.image.resize(dummy_data, shape=[2, 32, 32, 3], method='bilinear')
     
    init_out = model.init(init_key, sample_key, dummy_data[:2], dummy_label_init)
    init_params = init_out['params']
    init_sn = init_out['singular_vectors']
    learn_rate = CosineDecay(oargs.startlr, oargs.maxlr, oargs.minlr, oargs.warmup_steps, oargs.decay_steps)

    #create the train state. Only create an instance of EMATrainStateAccum if we need to, otherwise create a regular EMATrainState that does not store extra grads.
    mu_dtype = jnp.bfloat16 if oargs.mu_dtype=='bfloat16' else jnp.float32
    nu_dtype = jnp.bfloat16 if oargs.nu_dtype=='bfloat16' else jnp.float32
    optimizer = adam(learn_rate, b1=0.9, b2=0.9, mu_dtype=mu_dtype, nu_dtype=nu_dtype)
    print('Using accumulated optimizer with the following arguments:\n', oargs)

    
    if n_accums > 1:
        print(f"Using gradient accumulation with microbatch size of {dargs.batch_size}, real batch size of {dargs.batch_size*n_accums}, and {n_accums} gradient accumulations per update.")
        zero_grad = copy_pytree(init_params)
        zero_grad = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), zero_grad)
        state = EMATrainStateAccum.create(
            apply_fn=model.apply,
            params=init_params,
            ema_params=copy_pytree(init_params),
            tx=optimizer,
            ema_decay=oargs.ema_decay,
            singular_vectors=init_sn,
            current_grads=zero_grad
        )
    else:
        print(f"Using a batch size of {dargs.batch_size} without gradient accumulation.")
        state = EMATrainState.create(
            apply_fn=model.apply,
            params=init_params,
            ema_params=copy_pytree(init_params),
            tx=optimizer,
            ema_decay=oargs.ema_decay,
            singular_vectors=init_sn
        )


    #give some information about our model and dataset, then restore checkpoint
    print('Total Parameters', count_params(state.params))
    print("trying to restore checkpoint...")
    state = restore_checkpoint(state, targs.checkpoint_dirs[0])
    print("global step after restore:", int(state.step))
    state = flax.jax_utils.replicate(state, devices=devices)

    #make distributed train/sample fns, training rng, and metrics
    p_train_step = jax.pmap(
        fun=jax.jit(train_step),
        axis_name='shards',
    )
    
    rng = jax.random.PRNGKey(seed=0) #note, every time when restarting from preemption, this will use the same rng numbers

    loss = flax.jax_utils.replicate(jnp.float32(0), devices=devices) 
    pxz = flax.jax_utils.replicate(jnp.float32(0), devices=devices) 
    kl = flax.jax_utils.replicate(jnp.float32(0), devices=devices) 
    srloss = flax.jax_utils.replicate(jnp.float32(0), devices=devices) 
    gnorm = flax.jax_utils.replicate(jnp.float32(0), devices=devices) 
    skip_counter = flax.jax_utils.replicate(jnp.int32(0), devices=devices) 

    #train
    printl(f"starting/resuming training from step {int(unreplicate(state.step))}")
    s=time.time()
    for global_step, (train_inputs) in zip(range(int(unreplicate(state.step)), targs.iterations), dataset):
        # Train step
        rng, *train_step_rng = random.split(rng, num=jax.local_device_count() + 1)
        train_step_rng = jax.device_put_sharded(train_step_rng, devices)

        state,  loss, pxz, kl, gnorm, srloss, skip_counter = p_train_step(
            train_step_rng,
            state,
            train_inputs,
            
            loss, 
            pxz, 
            kl, 
            gnorm, 
            srloss, 
            skip_counter
        )

        if global_step % targs.log_freq==0: 
            a = unreplicate(loss)/targs.log_freq
            b = unreplicate(pxz)/targs.log_freq
            c = unreplicate(kl)/targs.log_freq
            d = unreplicate(gnorm)/(targs.log_freq*total_datadim)
            skips = unreplicate(skip_counter)
            results = f'Step: {unreplicate(state.step)}, Loss: {a}, Distortion: {b}, KL {c}, Gnorm {d}, Skips: {skips}, Time {round(time.time()-s)}s'  
            printl(results)

            loss *= 0
            pxz *= 0
            kl *= 0
            gnorm *= 0

        for checkpoint_dir, num_checkpoints, save_freq in zip(targs.checkpoint_dirs, targs.num_checkpoints, targs.save_freq):
            if global_step%save_freq==0:
                save_checkpoint(state, checkpoint_dir, unreplicate=True, keep=num_checkpoints)

        
        

if __name__ == '__main__':
    app.run(main)