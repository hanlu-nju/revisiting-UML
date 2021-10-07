from .mini_imagenet import MiniImageNet
from .cub import CUB
from .tiered_imagenet import TieredImageNet
from .cifarfs import CIFARFS
from .fc100 import FC100

dataset_dict = {
    'MiniImageNet': MiniImageNet,
    'CUB': CUB,
    'TieredImageNet': TieredImageNet,
    'CIFAR-FS': CIFARFS,
    'FC100': FC100
}
