import os.path as osp
import pickle

import numpy as np

from .base import BaseDataset


# THIS_PATH = osp.dirname(__file__)
# ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))


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


class FC100(BaseDataset):

    def __init__(self, setname, unsupervised, args, augment='none'):
        self.DATA_PATH = osp.join(args.data_root, 'FC100')
        super().__init__(setname, unsupervised, args, augment)

    @property
    def eval_setting(self):
        return [(5, 1), (5, 5), (5, 20), (5, 50)]

    def get_data(self, setname):
        data_train = load_data(osp.join(self.DATA_PATH, 'FC100_{}.pickle'.format(setname)))
        self.data = data_train['data']
        self.label = data_train['labels']

        # transform labels to 0,1,2,...
        _, self.label = np.unique(np.array(self.label), return_inverse=True)
        self.label = self.label.tolist()
        return self.data, self.label

    @property
    def image_size(self):
        return 32
