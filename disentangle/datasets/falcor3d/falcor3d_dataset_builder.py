import ipdb
import numpy as np
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    source_to_num_classes = {
        'lighting_intensity': 5,
        'lighting_x':     6,
        'lighting_y':     6,
        'lighting_z':     6,
        'camera_x': 6,
        'camera_y': 6,
        'camera_z': 6,
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
        path = dl_manager.download_and_extract('https://drive.google.com/uc?id=1XAQfFK1x6cpN1eiovbP0hVfLTm5SsSoJ')
        return {
            'train': self._generate_examples(path)
        }

    def _generate_examples(self, path):
        sources = np.load(path / 'Falcor3D_down128/train-rec.labels')
        source_names = [k for k in self.source_to_num_classes.keys()]
        source_values = [np.unique(sources[:, i]) for i in range(len(source_names))]
        for i, source in enumerate(sources):
            ret = {}
            ret['image'] = path / 'Falcor3D_down128' / 'images' / f'{i:06}.png'
            for j, value in enumerate(source):
                ret[f'value_{source_names[j]}'] = value
                ret[f'label_{source_names[j]}'] = np.where(source_values[j] == value)[0][0]
            yield i, ret
