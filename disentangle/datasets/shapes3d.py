import ipdb
import pathlib
import jax.numpy as jnp
import numpy as np
import resource
import einops

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
import tensorflow as tf
import tensorflow_datasets as tfds


def load(config):
    possible_dirs = config.data.possible_dirs
    while len(possible_dirs) > 0:
        possible_dir = pathlib.Path(possible_dirs.pop(0))
        try:
            train_set, info = tfds.load('shapes3d', split='train', data_dir=possible_dir, with_info=True)
            break
        except PermissionError as e:
            print(e)

    dataset_info = {}
    dataset_info['source_names'] = [
        k.replace('label_', '') for k in info.features.keys() if k.startswith('label_')
    ]
    dataset_info['num_sources'] = len(dataset_info['source_names'])
    label_names = [
        k for k in info.features.keys() if k.startswith('label_')
    ]
    dataset_info['num_train'] = info.splits['train'].num_examples
    sources_max = tf.cast(tf.stack([info.features[k].num_classes - 1 for k in label_names], axis=0), tf.float32)
    sources_min = tf.zeros_like(sources_max)

    def prepare_data(data):
        ret = {}

        x = einops.rearrange(data['image'], 'h w c -> c h w')
        x = tf.cast(x, tf.float32) / 255.
        x = x * 2 - 1  # [-1, 1]
        ret['x'] = x

        sources = tf.cast(tf.stack([data[k] for k in label_names], axis=0), tf.float32)
        sources = (sources - sources_min) / (sources_max - sources_min) * 2 - 1  # [-1, 1]
        ret['s'] = sources
        return ret

    val_set = train_set.take(config.data.num_val)
    train_set = train_set.shuffle(config.data.buffer_size, seed=config.data.seed, reshuffle_each_iteration=True) \
        .repeat() \
        .map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(config.data.batch_size) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    val_set = val_set.map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(config.data.batch_size) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)

    source_values = {
        'floor_hue':   np.linspace(0, 0.9, 10),
        'object_hue':  np.linspace(0, 0.9, 10),
        'orientation': np.linspace(-30, 30, 15),
        'scale':       np.linspace(0.75, 1.25, 8),
        'shape':       np.array([0, 1, 2, 3]),
        'wall_hue': np.linspace(0, 1, 10),
    }

    source_statistics = {}
    for k, v in source_values.items():
        source_statistics[k] = {
            'mean': v.mean(),
            'std':  v.std(),
        }

    source_mean = tf.convert_to_tensor(np.array([v['mean'] for v in source_statistics.values()]), dtype=tf.float32)
    source_std = tf.convert_to_tensor(np.array([v['std'] for v in source_statistics.values()]), dtype=tf.float32)

    return dataset_info, tfds.as_numpy(train_set), tfds.as_numpy(val_set)


if __name__ == '__main__':
    import omegaconf

    config = omegaconf.OmegaConf.create(
        {
            'data': {
                'possible_dirs': [
                    '/scr-ssd/kylehsu/data',
                    '/scr/kylehsu/data',
                    '/iris/u/kylehsu/data',
                ],
                'seed':          42,
                'num_val':       10000,
                'batch_size':    1024
            },
        }
    )
    _, train_set, val_set = load(config)
    tfds.benchmark(train_set, num_iter=1000, batch_size=config.data.batch_size)
