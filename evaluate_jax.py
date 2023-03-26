import jax
import jax.random as random
import jax.numpy as jnp
from optax import adam
from tensorflow.io import gfile
import numpy as np
import os
import flax
from tqdm.auto import tqdm
import time
import pickle

from absl import app, flags
from ml_collections.config_flags import config_flags

from jax_utils import restore_checkpoint
from model import VAE
from general_utils import save_images

args = flags.FLAGS
config_flags.DEFINE_config_file("config", "./config/imagenet256_kl8.py", "the location of the config path for the model. e.g. ./config/imagenet256_kl8.py.")
flags.DEFINE_string("save_dir", None, "the global directory you will save your results into.")
flags.DEFINE_string("checkpoint_path", None, "use this if loading a .p file (for the model params only)")
flags.DEFINE_string("checkpoint_dir", None, "use this if loading a flax checkpoint. specify both directory (with --checkpoint_dir) and step (with --step)")
flags.DEFINE_integer("step", -1, "use this if loading a flax checkpoint. specify both directory (with --checkpoint_dir) and step (with --step)")
flags.DEFINE_string("n_samples", "36", "the number of samples you want to create.")
flags.DEFINE_integer("nrow", 6, "if you are making a grid, the number of columns in the grid. By default, we use 6 columns.")
flags.DEFINE_integer("max_batch_size", 64, "the maximum allowable batch size for sampling.")
flags.DEFINE_string("label", "-1", "If the model is class conditional, generates images from a certain class. Set to -1 for randomly chosen classes. Set to the number of classes in your dataset (e.g. 1000 for imagenet) for unconditional sampling.")
flags.DEFINE_string("mean_scale", "1.5", "the weight for classifier-free guidance of the mean. can be comma-separated list, corresponding to guidance strength at each resolution")
flags.DEFINE_string("var_scale", "3.0", "the weight for classifier-free guidance of the variance. can be comma-separated list, corresponding to guidance strength at each resolution")
flags.DEFINE_integer("seed", 0, "seed for PRNGKey")
flags.mark_flags_as_required(["config", "save_dir"])



def main(_):
    if not gfile.isdir(args.save_dir):
        gfile.makedirs(args.save_dir)

    config = args.config
    config.unlock()
    margs = config.model
    model = VAE(**margs)
    

    if args.checkpoint_path is not None:
        with open(args.checkpoint_path, 'rb') as f:
            state_dict = pickle.load(f)
        
        if "ema_params" in state_dict.keys(): params = state_dict["ema_params"]
        elif "params" in state_dict.keys(): params = state_dict["params"]
        else: raise AssertionError("Either params or ema_params must be a key in statedict")
    else: 
        step = args.step if args.step != -1 else None
        state_dict = restore_checkpoint(None, ckpt_dir=args.checkpoint_dir, step=step)
        
        if "ema_params" in state_dict.keys(): params = state_dict["ema_params"]
        elif "params" in state_dict.keys(): params = state_dict["params"]
        else: raise AssertionError("Either params or ema_params must be a key in statedict")
    
    devices = jax.devices()
    params = flax.jax_utils.replicate(params, devices=devices)

    res = config.model.resolution
    n_samples = int(args.n_samples)
    label = int(args.label)
    if "," in args.mean_scale: mean_scale = [float(l) for l in args.mean_scale.split(",")]
    else: mean_scale = float(args.mean_scale)
    if "," in args.var_scale: var_scale = [float(l) for l in args.var_scale.split(",")]
    else: var_scale = float(args.var_scale)

    def gen_func(batch_label, rng, params):
        vae = VAE(**dict(margs))
        return vae.apply({'params': params}, rng=rng, num=len(batch_label), 
            label=batch_label, method=model.p_sample, mutable=['singular_vectors'], mweight=mean_scale, vweight=var_scale
        )
    
    p_gen_func = jax.pmap(
        fun=jax.jit(gen_func),
        axis_name='shards',
    )

    samples = np.zeros([0, res, res, 4]).astype('float32')
    rng = random.PRNGKey(args.seed)
    rng, *eval_rng = random.split(rng, num=len(devices) + 1)
    eval_rng = jax.device_put_sharded(eval_rng, devices)

    print(f"creating {n_samples} of label {label} with mean and var scales of {mean_scale} and {var_scale}")
    for n in tqdm(range(0, n_samples, args.max_batch_size)):            
        if not model.num_classes:
            batch_label = None
        elif label == -1:
            rng, label_gen_key = random.split(rng)
            batch_label = random.randint(label_gen_key, [len(devices), args.max_batch_size//len(devices)], minval=0, maxval=config.model.num_classes)
        else:
            batch_label = jnp.reshape(
                jnp.array([label] * args.max_batch_size, dtype=jnp.int32), (len(devices), args.max_batch_size//len(devices))
            )
        
        mweight = flax.jax_utils.replicate(jnp.float32(mean_scale), devices=devices)
        vweight = flax.jax_utils.replicate(jnp.float32(var_scale), devices=devices)

        current_images = p_gen_func(batch_label, eval_rng, params)
        current_images, _ = current_images
        current_images = jnp.reshape(jax.device_get(current_images), (-1, res, res, 4))
        samples = np.concatenate((samples, np.array(current_images)), axis=0)

    print("samples shape:", samples.shape)
    samples = samples[:n_samples]
    
    if model.num_classes == 0 or label == model.num_classes:
        label_string = "uncond"
    elif label == -1:
        label_string = f"random_classes_m{mean_scale}_v{var_scale}"
    else:
        label_string = f"class_{str(label)}_m{mean_scale}_v{var_scale}"

    samples_identifier = f"{len(gfile.glob(f'{args.save_dir}/*.npz'))}_{label_string}"
    samples_path = os.path.join(args.save_dir, f"samples_{samples_identifier}.npz")
    
    if not gfile.exists(args.save_dir):
        gfile.mkdir(args.save_dir)

    
    np.savez("tmpfile.npz", arr0=samples)
    gfile.copy("tmpfile.npz", samples_path)
    time.sleep(3.0)
    gfile.remove("tmpfile.npz")
    print(f"Saved {len(samples)} samples to {samples_path}")

if __name__ == '__main__':
    app.run(main)
