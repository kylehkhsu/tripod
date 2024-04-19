import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')  # don't let tf hog GPU

from . import shapes3d, isaac3d, mpi3d, falcor3d


def load(config):
    match config.data.name:
        case 'shapes3d':
            dataset_info, train_set, val_set = shapes3d.load(config)
        case 'isaac3d':
            dataset_info, train_set, val_set = isaac3d.load(config)
        case 'falcor3d':
            dataset_info, train_set, val_set = falcor3d.load(config)
        case 'mpi3d':
            dataset_info, train_set, val_set = mpi3d.load(config)
        case _:
            raise ValueError(f'unknown dataset {config.data.name}')
    return dataset_info, train_set, val_set

