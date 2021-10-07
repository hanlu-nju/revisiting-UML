from __future__ import print_function

import os
import pickle
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, make_dataset, IMG_EXTENSIONS
from .base import BaseDataset
from PIL import Image
import numpy as np


def search_dir_or_file(dirs, description='data directory'):
    found = None

    for d in dirs:
        if os.path.exists(d):
            found = d
            break
    if found is None:
        raise FileNotFoundError(f'{description} not found')
    print(f'{description} : {found}')
    return found


# Set the appropriate paths of the datasets here.
POSSIBLE_DATA_DIR = [
    '/home/hanlu/data/tiered_imagenet_raw',
    '/data/hanlu/tiered_imagenet_raw',
    '/data/yangy/tieredimagenet',
    './datasets/tieredimagenet',
]


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data


# file_path = {'train':[os.path.join(IMAGE_PATH, 'train_images.npz'), os.path.join(IMAGE_PATH, 'train_labels.pkl')],
#              'val':[os.path.join(IMAGE_PATH, 'val_images.npz'), os.path.join(IMAGE_PATH,'val_labels.pkl')],
#              'test':[os.path.join(IMAGE_PATH, 'test_images.npz'), os.path.join(IMAGE_PATH, 'test_labels.pkl')]}

class TieredImageNet(BaseDataset):

    def get_data(self, setname):
        root = search_dir_or_file([os.path.join(d, setname) for d in POSSIBLE_DATA_DIR])
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS, None)
        # self.samples = samples
        label = [s[1] for s in samples]
        data = [s[0] for s in samples]
        return data, label

    def get_n_classes(self):
        return len(self.classes)

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        if isinstance(data, str):
            data = Image.open(data).convert('RGB')
        elif isinstance(data, np.ndarray):
            data = Image.fromarray(data)
        if self.unsupervised:
            image_list = []
            # inp = self.flip_lr(data)
            if self.augment == 'AMDIM':
                data = self.flip_lr(data)
            for _ in range(self.repeat):
                image_list.append(self.transform(data))
                # inp = self.flip_lr(Image.open(data).convert('RGB'))
            return image_list, label
        else:
            image = self.transform(data)
        return image, label

    def __len__(self):
        return len(self.data)

    @property
    def eval_setting(self):
        return [(5, 1), (5, 5), (5, 20), (5, 50)]

    @property
    def image_size(self):
        return 84
