import ipdb
import numpy as np
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    source_to_num_classes = {
        'object_color': 4,
        'object_shape': 4,
        'object_size': 2,
        'camera_height': 3,
        'background_color': 3,
        'horizontal_axis': 40,
        'vertical_axis': 40,
    }

    def _info(self) -> tfds.core.DatasetInfo:

        features = {}
        for source, num_classes in self.source_to_num_classes.items():
            features[f'label_{source}'] = tfds.features.ClassLabel(num_classes=num_classes)
            # features[f'value_{source}'] = tfds.features.Tensor(shape=[], dtype=np.float32)
        features['image'] = tfds.features.Image(shape=(64, 64, 3))

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(features),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.download('https://drive.google.com/uc?id=1Tp8eTdHxgUMtsZv5uAoYAbJR1BOa_OQm')
        return {
            'train': self._generate_examples(path)
        }

    def _generate_examples(self, path):
        images = np.load(path)['images']
        source_sizes = list(self.source_to_num_classes.values())
        source_names = list(self.source_to_num_classes.keys())

        for i, image in enumerate(images):
            coordinates = np.unravel_index(i, source_sizes)
            ret = {}
            ret['image'] = image
            for j, label in enumerate(coordinates):
                ret[f'label_{source_names[j]}'] = label
            yield i, ret