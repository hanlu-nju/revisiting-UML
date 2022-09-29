import os.path as osp

from tqdm import tqdm

from model.dataloader.base import BaseDataset, ROOT_DIRS, search_dir

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..', '..'))
data_dirs = [
    'miniimagenet',
    'mini-imagenet',
]


class MiniImageNet(BaseDataset):
    """ Usage:
    """

    def __init__(self, setname, unsupervised, args, augment='none'):
        self.IMAGE_PATH1 = search_dir([osp.join(args.data_root, di, 'images') for di in data_dirs])
        self.SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split')
        self.CACHE_PATH = osp.join(ROOT_PATH, '.cache/')
        super().__init__(setname, unsupervised, args, augment)

    @property
    def image_size(self):
        return 84

    @property
    def eval_setting(self):
        if hasattr(self.args, 'short') and self.args.short:
            return [(5, 1), (5, 5)]
        return [(5, 1), (5, 5),
                # (5, 10),
                (5, 20),
                # (5, 30),
                (5, 50)]

    @property
    def split_path(self):
        return self.SPLIT_PATH

    @property
    def cache_path(self):
        return self.CACHE_PATH

    def parse_csv(self, csv_path):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH1, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        return data, label
