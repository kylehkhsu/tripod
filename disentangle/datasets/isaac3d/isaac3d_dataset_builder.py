import ipdb
import numpy as np
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    source_to_num_classes = {
        'object_shape':       3,
        'robot_x':            8,
        'robot_y':            5,
        'camera_height':      4,
        'object_scale':       4,
        'lighting_intensity': 4,
        'lighting_y_dir':     6,
        'object_color':       4,
        'wall_color':         4,
    }

    def _info(self) -> tfds.core.DatasetInfo:

        features = {}
        for source, num_classes in self.source_to_num_classes.items():
            features[f'label_{source}'] = tfds.features.ClassLabel(num_classes=num_classes)
            features[f'value_{source}'] = tfds.features.Tensor(shape=[], dtype=np.float32)
        features['image'] = tfds.features.Image(shape=(128, 128, 3))

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(features),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.download_and_extract('https://drive.google.com/uc?id=1OmQ1G2wnm6eTsSFGTKFZZAh5D3nQTW1B')
        return {
            'train': self._generate_examples(path)
        }

    def _generate_examples(self, path):
        sources = np.load(path / 'Isaac3D_down128/labels.npy')
        source_names = [k for k in self.source_to_num_classes.keys()]
        source_values = [np.unique(sources[:, i]) for i in range(len(source_names))]
        for i, source in enumerate(sources):
            ret = {}
            ret['image'] = path / 'Isaac3D_down128' / 'images' / f'{i:06}.png'
            for j, value in enumerate(source):
                ret[f'value_{source_names[j]}'] = value
                ret[f'label_{source_names[j]}'] = np.where(source_values[j] == value)[0][0]
            yield i, ret
