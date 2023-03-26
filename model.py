import jax
import jax.numpy as jnp
import numpy as np

from typing import Iterable, Any
from flax import linen as nn
from flax.linen import initializers 

from jax_utils import get_smoothed_variance, sample_mvn_deterministic, sample_diag_mvn, RngGen, compute_mvn_kl
from general_utils import get_resolutions, get_evenly_spaced_indices
from nn import Conv2D, Downsample, Upsample, ResBlockFFN, ResBlockNoFFN, AttentionLayer

MIN_REMAT_RESO = 8 #inclusive

class StochasticConvLayer(nn.Module):
    c: int
    zdim: int
    resolution: int
    w_scale: float=1.0
    num_classes: int=0
    dff_mul: int=4
    h_prior: bool=False

    def setup(self):
        c = self.c
        zdim = self.zdim
        w_scale = self.w_scale
        sr_lam = float(self.resolution)
        is_conv = self.resolution>=4
        dff_mul = self.dff_mul

        ResBlock = ResBlockFFN if self.dff_mul else ResBlockNoFFN
        if self.resolution >= MIN_REMAT_RESO:
            block = nn.remat(ResBlock)
        else:
            block = ResBlock

        pchannels = c+zdim*2 if self.h_prior else c
        self.prior_block = block(c, pchannels, w_scale=1e-10, sr_lam=sr_lam, num_classes=self.num_classes, is_conv=is_conv, dff_mul=dff_mul)
        self.posterior_block = block(c, zdim*2, w_scale=1.0, sr_lam=sr_lam, num_classes=self.num_classes, is_conv=is_conv, dff_mul=dff_mul)
        self.actsproj = Conv2D(self.c, kernel_size=1, w_scale=1.0)
        self.zproj = Conv2D(self.c, kernel_size=1, w_scale=w_scale, use_bias=False)
        self.shared_block = block(c, c, w_scale=w_scale, num_classes=self.num_classes, is_conv=is_conv, dff_mul=dff_mul)
        
    def p_sample(self, rng, x_c, x_u=None, label=None, uncond_label=None, mweight=0.0, vweight=0.0, T=1.):
        zdim = self.zdim
        is_guided = (x_u is not None and self.h_prior)

        if self.h_prior:
            p_out_c, _ = self.prior_block(x_c, label)
            pmean_c, pv_unconstrained_c, h_c = p_out_c[..., :zdim], p_out_c[..., zdim:zdim*2], p_out_c[..., zdim*2:]
            pvar_c = get_smoothed_variance(pv_unconstrained_c)
        
            if is_guided:
                p_out_u, _ = self.prior_block(x_u, uncond_label)
                pmean_u, pv_unconstrained_u, h_u = p_out_u[..., :zdim], p_out_u[..., zdim:zdim*2], p_out_u[..., zdim*2:]
                pvar_u = get_smoothed_variance(pv_unconstrained_u)
                pmean = (pmean_c + mweight * (pmean_c - pmean_u))
                logvar_c, logvar_u = jnp.log(pvar_c) , jnp.log(pvar_u)
                pvar = pvar_c * jnp.exp(vweight * (logvar_c - logvar_u))
            else:
                pmean, pvar = pmean_c, pvar_c
        else:
            pmean = jnp.zeros((x_c.shape[0], x_c.shape[1], x_c.shape[2], self.zdim))
            pvar = jnp.ones((x_c.shape[0], x_c.shape[1], x_c.shape[2], self.zdim))
            h_c, _ = self.prior_block(x_c, label)

        z = sample_diag_mvn(rng, pmean, pvar, temp=T)
        z = self.zproj(z)
        
        x_c += (z + h_c)
        x_c = self.shared_block(x_c, label)
        
        if is_guided:
            x_u += (z + h_u)
            x_u = self.shared_block(x_u, uncond_label)

        return x_c, x_u

    
    def __call__(self, eps, x, acts, label=None):
        if label is not None: label, cf_guidance_label = label #q is always conditional, p is not necessarily
        else: cf_guidance_label = None
        zdim = self.zdim

        concatted = jnp.concatenate((x, acts), axis=-1)
        x_and_acts = self.actsproj(concatted)
        q_out, sr_loss_q = self.posterior_block(x_and_acts, label)
        qmean, qv_unconstrained = jnp.split(q_out, 2, axis=-1) 
        qvar = get_smoothed_variance(qv_unconstrained)    

        p_out, sr_loss_p = self.prior_block(x, cf_guidance_label)
        if self.h_prior:
            pmean, pv_unconstrained, h = p_out[..., :zdim], p_out[..., zdim:zdim*2], p_out[..., zdim*2:]
            pvar = get_smoothed_variance(pv_unconstrained)
        else:
            h = p_out
            sr_loss_p *= 0.
            pmean = jnp.zeros((x.shape[0], x.shape[1], x.shape[2], self.zdim))
            pvar = jnp.ones((x.shape[0], x.shape[1], x.shape[2], self.zdim))
        kl_unweighted = compute_mvn_kl(qmean, qvar, pmean, pvar)

        z = sample_mvn_deterministic(eps, qmean, qvar)  
        z = self.zproj(z)
        x += (z + h)
        x = self.shared_block(x, cf_guidance_label)
        return x, kl_unweighted, sr_loss_p+sr_loss_q

class DecoderLevel(nn.Module):
    c: int
    zdim: int
    nlayers: int
    w_scale: float
    num_classes: int
    num_attention: int
    current_resolution: int
    max_resolution: int
    c_next: Any
    dff_mul: int
    head_count: int
    h_prior: bool

    def setup(self):
        is_conv = (self.current_resolution >= 4)
        layer_list = []
        attention_indices = get_evenly_spaced_indices(self.nlayers, self.num_attention)

        for i in range(1, self.nlayers+1):

            if self.current_resolution >= MIN_REMAT_RESO:
                convlayer = nn.remat(StochasticConvLayer)
                attlayer = nn.remat(AttentionLayer)
            else:
                convlayer = StochasticConvLayer
                attlayer = AttentionLayer

            layer_list.append(
                convlayer(
                    c=self.c, 
                    zdim=self.zdim, 
                    resolution=self.current_resolution,
                    w_scale=self.w_scale, 
                    num_classes=self.num_classes,
                    dff_mul=self.dff_mul,
                    h_prior=self.h_prior
                )
            )
            if i in attention_indices:
                layer_list.append(attlayer(self.c, self.head_count))
        
        if self.current_resolution < self.max_resolution:
            strides = 2 if is_conv else 4
            layer_list.append(Upsample(self.c_next, strides))
        
        self.layer_list = layer_list

    def p_sample(self, rng, x_c, x_u=None, label=None, uncond_label=None, mweight=0.0, vweight=0.0, T=1.0):
        for layer in self.layer_list:
            if isinstance(layer, StochasticConvLayer):
                x_c, x_u = layer.p_sample(rng, x_c, x_u, label, uncond_label, mweight=mweight, vweight=vweight, T=T)
            else:
                x_c = layer(x_c)
                if x_u is not None:
                    x_u = layer(x_u)

        return x_c, x_u

    def __call__(self, rng, x, acts, label=None):
        shape = (x.shape[0], self.current_resolution, self.current_resolution, self.zdim)

        KLs = []
        SR_Losses = 0.
        i = 0

        for layer in self.layer_list:
            if isinstance(layer, StochasticConvLayer):
                eps = jax.random.normal(next(rng), shape=shape)
                x, kl, sr_loss = layer(eps, x, acts, label)
                KLs.append(kl)
                SR_Losses += sr_loss
                i += 1
            else:
                x = layer(x)

        return x, KLs, SR_Losses

class EncoderLevel(nn.Module):
    c: int
    nlayers: int
    current_resolution: int
    c_next: Any
    num_attention: int=0
    w_scale: float=1.
    num_classes: int=0
    dff_mul: int=4
    head_count: int=4

    def setup(self):
        nlayers = self.nlayers
        is_conv = (self.current_resolution >= 4)

        layer_list = []
        attention_indices = get_evenly_spaced_indices(nlayers, self.num_attention)

        ResBlock = ResBlockFFN if self.dff_mul else ResBlockNoFFN
        if self.current_resolution >= MIN_REMAT_RESO:
            block = nn.remat(ResBlock)
        else:
            block = ResBlock
        for i in range(nlayers):
            layer_list.append(
                block(
                    self.c, 
                    self.c, 
                    self.w_scale, 
                    num_classes=self.num_classes, 
                    is_conv=is_conv,
                    dff_mul=self.dff_mul
                )
            )
            
            if i in attention_indices:
                if self.current_resolution >= MIN_REMAT_RESO:
                    attlayer = nn.remat(AttentionLayer)
                else:
                    attlayer = AttentionLayer
                
                layer_list.append(attlayer(self.c, self.head_count))
            
        if self.current_resolution > 1:
            strides = 2 if self.current_resolution > 4 else 4
            layer_list.append(Downsample(self.c_next, strides)) #note: c_next might be None
        self.layer_list = layer_list

    def __call__(self, x, label=None):
        acts = None
        for layer in self.layer_list:
            if isinstance(layer, Downsample):
                acts = x
            x = layer(x, label)

        if acts is None: acts = x
        return x, acts

        
class Encoder(nn.Module):
    c: int
    c_mult: Iterable[int]
    nlayers: Iterable[int]
    resolution: int
    num_attention: Iterable[int]
    num_classes: int
    dff_mult: Iterable[int]
    num_attention_heads: Iterable[int]

    @nn.compact
    def __call__(self, img, label=None):
        C = [self.c * mult for mult in self.c_mult]
        nlayers = self.nlayers
        w_scale = 1/jnp.sqrt(sum(nlayers))
        num_resolutions = len(nlayers)
        resolutions = get_resolutions(self.resolution, num_resolutions)

        x = Conv2D(C[0], kernel_size=3)(img)
        acts = []
        for i in range(num_resolutions):
            c_next = C[i+1] if i<num_resolutions-1 else None
            if c_next == C[i]: c_next = None

            x, acts_i = EncoderLevel(
                C[i],
                nlayers[i],
                resolutions[i],
                c_next,
                self.num_attention[i],
                w_scale,
                self.num_classes,
                self.dff_mult[i],
                self.num_attention_heads[i]
            )(x, label)
            acts.append(acts_i)
        return acts

class SuperResolutionVAE(nn.Module):
    c: int
    c_enc: int
    c_mult: Iterable[int]
    zdim: int
    nlayers: Iterable[int]
    enc_nlayers: Iterable[int]
    num_attention: Iterable[int]
    resolution: int
    downfactor: int
    dff_mult: Iterable[int]
    num_attention_heads: Iterable[int]

    def setup(self):
        self.encoder = Encoder(self.c_enc, self.c_mult, self.enc_nlayers, self.resolution, self.num_attention, 0, self.dff_mult, self.num_attention_heads)

        layer_list = []
        resolutions = get_resolutions(self.resolution, len(self.nlayers))
        C = [self.c * mult for mult in self.c_mult]
        w_scale = 1/np.sqrt(sum(self.nlayers))
        self.inproj = Conv2D(C[-1], 1)        

        for i in reversed(range(len(resolutions))):
            c_next = C[i-1] if i>0 else None
            if c_next == C[i]: c_next = None

            layer_list.append(
                DecoderLevel(
                    c=C[i],
                    zdim=self.zdim,
                    nlayers=self.nlayers[i],
                    w_scale=w_scale,
                    num_classes=0,
                    num_attention=self.num_attention[i],
                    current_resolution=resolutions[i],
                    max_resolution=resolutions[0],
                    c_next=c_next,
                    dff_mul=self.dff_mult[i],
                    head_count=self.num_attention_heads[i],
                    h_prior=False
                )
            )

        self.layer_list = layer_list
        self.outproj = Conv2D(6, 1)

    def p_sample(self, rng, imglr):
        rng = RngGen(rng)
        x = self.inproj(imglr)

        for decoderlevel in self.layer_list:
            x, _ = decoderlevel.p_sample(rng, x)

        x = self.outproj(x)
        return x[..., :3]

    def __call__(self, rng, img, label=None):
        B, H, W, C = img.shape
        imglr = jax.image.resize(img, shape=[B, H // self.downfactor, W // self.downfactor, C], method='bilinear')
        rng = RngGen(rng)
        
        x = self.inproj(imglr)
        acts = self.encoder(img)
        KLs = []
        SR_Losses = 0.

        for i, decoderlevel in enumerate(self.layer_list):
            j = -i+len(self.nlayers)-1
            x, kls, sr_losses = decoderlevel(rng, x, acts[j])
            KLs.extend(kls)
            SR_Losses += sr_losses
                
        x = self.outproj(x)
        return x, KLs, SR_Losses



class VAE(nn.Module):
    resolution: int
    c: int
    c_enc: int
    c_mult: Iterable[int]
    datadim: int
    zdim: int
    nlayers: Iterable[int]
    enc_nlayers: Iterable[int]
    num_attention: Iterable[int]
    dff_mult: Iterable[int]
    num_attention_heads: Iterable[int]
    num_classes: int
    h_prior: bool

    def setup(self):
        resolutions = get_resolutions(self.resolution, len(self.nlayers))
        C = [self.c * mult for mult in self.c_mult]
        w_scale = 1/np.sqrt(sum(self.nlayers))

        self.cond = self.num_classes > 0
        self.embed = nn.Embed(self.num_classes+1, C[-1])
        self.encoder = Encoder(self.c_enc, self.c_mult, self.enc_nlayers, self.resolution, self.num_attention, self.num_classes, self.dff_mult, self.num_attention_heads)
        self.initial_x = self.param('initial_x', initializers.zeros, (1, resolutions[-1], resolutions[-1], C[-1]))
        

        layer_list = []
        for i in reversed(range(len(resolutions))):
            c_next = C[i-1] if i>0 else None
            if c_next == C[i]: c_next = None

            layer_list.append(
                DecoderLevel(
                    c=C[i],
                    zdim=self.zdim,
                    nlayers=self.nlayers[i],
                    w_scale=w_scale,
                    num_classes=self.num_classes,
                    num_attention=self.num_attention[i],
                    current_resolution=resolutions[i],
                    max_resolution=resolutions[0],
                    c_next=c_next,
                    dff_mul=self.dff_mult[i],
                    head_count=self.num_attention_heads[i],
                    h_prior=self.h_prior
                )
            )

        self.layer_list = layer_list
        self.outproj = Conv2D(self.datadim*2, 1)

    def p_sample(self, rng, num=10, label=None, mweight=0.0, vweight=0.0):
        rng = RngGen(rng)
        if self.num_classes:
            uncond_label = jnp.full_like(label, self.num_classes)
            label = self.embed(label)
            uncond_label = self.embed(uncond_label)
        else:
            uncond_label = None
            
        is_guided = (mweight != 0 or vweight != 0)
        x_c = jnp.tile(self.initial_x, [num, 1, 1, 1])
        x_u = jnp.tile(self.initial_x, [num, 1, 1, 1]) if is_guided else None

        if isinstance(mweight, float): mweight = [mweight] * len(self.nlayers)
        if isinstance(vweight, float): vweight = [vweight] * len(self.nlayers)

        for i, decoderlevel in enumerate(self.layer_list):
            x_c, x_u = decoderlevel.p_sample(rng, x_c, x_u, label, uncond_label, mweight=mweight[i], vweight=vweight[i])

        x_c = self.outproj(x_c)
        return x_c[..., :self.datadim]
    
    def __call__(self, rng, img, label=None):
        rng = RngGen(rng)
        if self.cond:  
            uncond_label = jnp.full_like(label, self.num_classes)
            mask = jnp.greater(jax.random.uniform(next(rng), label.shape), 0.9)
            cf_guidance_label = (label*(1-mask) + mask*uncond_label).astype(jnp.int32)
            
            label = self.embed(label)
            cf_guidance_label = self.embed(cf_guidance_label)
            generator_label = (label, cf_guidance_label)
        else:
            label, generator_label = None, None
        
        
        x = jnp.tile(self.initial_x, [img.shape[0], 1, 1, 1])
        acts = self.encoder(img, label=label)
        KLs = []
        SR_Losses = 0.

        for i, decoderlevel in enumerate(self.layer_list):
            j = -i+len(self.nlayers)-1
            x, kls, sr_losses = decoderlevel(rng, x, acts[j], generator_label)
            KLs.extend(kls)
            SR_Losses += sr_losses
                
        x = self.outproj(x)
        return x, KLs, SR_Losses