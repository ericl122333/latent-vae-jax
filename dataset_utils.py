import tensorflow_datasets as tfds
import tensorflow as tf
import jax
import flax

import random as python_random
import os
import functools

def mapfn(example, dargs):
    features = {
        "x": tf.io.FixedLenFeature([], tf.string)
    }
    if dargs.num_classes > 0:
        features["y"] =  tf.io.FixedLenFeature([], tf.int64)
    example = tf.io.parse_single_example(example, features)

    #image = tf.io.parse_tensor(example["x"], tf.float16)
    if dargs.dtype=='float32': dtype = tf.float32
    elif dargs.dtype=='float16': dtype = tf.float16
    else: dtype = tf.uint8

    image = tf.io.parse_tensor(example["x"], dtype)
    image = (tf.cast(image, tf.float32) - dargs.shift) / dargs.scale
    image = tf.clip_by_value(image, -4.0, 4.0)
    #print(dargs.dataset_name, dargs.resolution)
    if dargs.dataset_name == "bedroom256":
        image = tf.transpose(image, perm=[1, 2, 0])   
    #image = tf.reshape(image, [dargs.resolution, dargs.resolution, dargs.channels])



    data_entries = [image]
    if dargs.num_classes > 0:
        label = tf.cast(example["y"], dtype=tf.int32)
        data_entries.append(label)
    else:
        data_entries.append(None)
    
    return tuple(data_entries)
    
def load_and_shard_tf_batch(xs, global_batch_size):
    local_device_count = jax.local_device_count()
    def _prepare(x):
        return x.reshape((local_device_count, global_batch_size // local_device_count) + x.shape[1:])
    return jax.tree_map(_prepare, xs)

def tfds_to_jax_dataset(dataset, batch_size):
    dataset = tfds.as_numpy(dataset)
    dataset = map(lambda x: load_and_shard_tf_batch(x, batch_size), dataset)
    dataset = flax.jax_utils.prefetch_to_device(dataset, 1) #one is probably okay? info here: https://flax.readthedocs.io/en/latest/api_reference/flax.jax_utils.html says 
    return dataset

def create_dataset(dargs):
    tfrecord_filenames = [os.path.join(dargs.data_dir, fname) for fname in tf.io.gfile.listdir(dargs.data_dir)]
    python_random.shuffle(tfrecord_filenames) #mixes up tfrecord file orders so the first files dont get seen more frequently than others when using preemptible instances.

    raw_dataset = tf.data.TFRecordDataset(tfrecord_filenames, num_parallel_reads=tf.data.AUTOTUNE)
    raw_dataset = raw_dataset.map(
        functools.partial(mapfn, dargs=dargs),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    raw_dataset = raw_dataset.shuffle(dargs.batch_size*10).batch(dargs.batch_size, drop_remainder=True)
    raw_dataset = raw_dataset.repeat().prefetch(tf.data.AUTOTUNE)
    
    dataset = tfds_to_jax_dataset(raw_dataset, dargs.batch_size)
    return dataset
