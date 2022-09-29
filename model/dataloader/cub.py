import os.path as osp

from model.dataloader.base import BaseDataset
from .base import ROOT_DIRS, search_dir

THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))


# dirs = ['/home/yehj/Few-Shot/data/cub/images',
#         '/home/amax/data/cub/images',
#         '/home/hanlu/data/cub/images']

# IMAGE_PATH = search_dir(dirs)


# This is for the CUB dataset
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)


class CUB(BaseDataset):

    def __init__(self, setname, unsupervised, args, augment='none'):
        super().__init__(setname, unsupervised, args, augment)
        self.IMAGE_PATH = osp.join(args.data_root, 'cub/images')
        self.SPLIT_PATH = osp.join(ROOT_PATH2, 'data/cub/split')
        self.CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')

    @property
    def image_size(self):
        return 84

    @property
    def eval_setting(self):
        return [(5, 1), (5, 5), (5, 20)]

    @property
    def split_path(self):
        return self.SPLIT_PATH

    @property
    def cache_path(self):
        return self.CACHE_PATH

    def parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            label.append(lb)

        return data, label
