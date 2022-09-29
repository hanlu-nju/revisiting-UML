from __future__ import print_function
import os
import shutil
import time
import pprint
import torch
import argparse
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from model.dataloader import dataset_dict

import torch
import torch.nn as nn


def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W - cut_w + 1)
    cy = np.random.randint(H - cut_h + 1)

    bbx1 = np.clip(cx, 0, W)
    bby1 = np.clip(cy, 0, H)
    bbx2 = np.clip(cx + cut_w, 0, W)
    bby2 = np.clip(cy + cut_h, 0, H)

    return bbx1, bby1, bbx2, bby2


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(dir_path, scripts_to_save=None):
    if os.path.exists(dir_path):
        if input('{} exists, remove? ([y]/n)'.format(dir_path)) != 'n':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print('copy {} to {}'.format(src_file, dst_file))
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)


class Averager:

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x, n=1):
        self.v = (self.v * self.n + x) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

    def reset(self):
        self.n = 0
        self.v = 0


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item() * 100
    else:
        return (pred == label).type(torch.FloatTensor).mean().item() * 100


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


def normalized_dot(proto, query, type='sns'):
    '''
    :param type: distance type
    :param proto: (num_way,emb_dim)
    :param query: (num_query,num_way,emb_dim)
    :param max_pool:
    :return:
    '''

    if type == 'sns':
        proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
    elif type == 'cosine':
        proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
        query = F.normalize(query, dim=-1)
    else:
        assert type == 'dot'
    logits = torch.mm(query, proto.t())
    return logits


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def postprocess_args(args, train=True):
    if args.eval_dataset is None:
        args.eval_dataset = args.dataset
    model_name = args.model_class
    if train:
        save_path1 = '-'.join(
            [args.dataset, args.eval_dataset, model_name, args.backbone_class,
             '{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query), args.additional])

        save_path2 = '_'.join([str('_'.join(args.step_size.split(','))), str(args.gamma),
                               'lars{}'.format(args.lars),
                               'lr{:.2g}mul{:.2g}'.format(args.lr, args.lr_mul),
                               str(args.lr_scheduler),
                               'T1{}T2{}'.format(args.temperature, args.temperature2),
                               'b{}'.format(args.balance),
                               'bsz{:03d}'.format(max(args.way, args.num_classes) * (args.shot + args.query)),
                               'batch{:03d}'.format(args.batch_size),
                               'ntask{:03d}'.format(args.num_tasks),
                               'nclass{:03d}'.format(args.num_classes),
                               'ep{}'.format(args.max_epoch),
                               'eval{}'.format(args.eval)
                               ])

        if args.unsupervised:
            save_path1 = 'uml_' + save_path1

        if args.init_weights is not None:
            f_path = args.init_weights.strip().split('/')[-1]
            save_path1 += '-Pre_%s' % (f_path[:f_path.rfind('.')])
        save_path1 += '-DIS_%s' % args.similarity

        if args.additional == 'HMS':
            save_path1 += f'_negs{args.hard_negs}_s{args.strength}'

        if args.augment == 'randaug':
            augment_str = f'{args.augment}_{args.n}_{args.m}'
        else:
            augment_str = args.augment
        save_path2 += '-Aug_%s' % augment_str

        if args.model_class == 'TSPHead':
            save_path2 = 'h{}_l{}_do{}_d{}_'.format(args.t_heads, args.t_layers,
                                                    args.t_dropout, args.t_dim) + save_path2

    else:
        save_path1 = 'eval-%s' % ('-'.join([args.eval_dataset, model_name, args.backbone_class,
                                            args.additional, args.ps]))
        save_path2 = '-'.join(['evalall_{}'.format(args.eval_all),
                               '{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query), ])
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    os.makedirs(args.save_path, exist_ok=True)
    args.filename = os.path.join(save_path1, save_path2)
    return args


def get_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--num_test_episodes', type=int, default=10000)
    parser.add_argument('--model_class', type=str, default='ProtoNet',
                        # choices=['ProtoNet'，]
                        )
    parser.add_argument('--similarity', type=str, default='sns', choices=['dot', 'sns', 'cosine', 'euclidean'])

    parser.add_argument('--backbone_class', type=str, default='Res12',
                        choices=['ConvNet', 'Res12', 'Res18', 'WRN']
                        )
    parser.add_argument('--dropblock_size', type=int, default=2)

    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'CUB', 'FC100', 'CIFAR-FS'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--class_number', type=int, default=64)

    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--unsupervised', action='store_true', default=False)
    parser.add_argument('--additional', type=str, default='none',
                        choices=['none', 'HMS'])
    parser.add_argument('--eval_all', action='store_true', default=False)
    parser.add_argument('--lars', action='store_true', default=False)
    parser.add_argument('--num_fc', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=3)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--balance', type=float, default=1)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--temperature2', type=float, default=1)  # the temperature in the

    # params for HMS
    parser.add_argument('--hard_negs', type=int, default=10)
    parser.add_argument('--strength', type=float, default=0.5)

    # param for TSP-Head
    parser.add_argument('--t_heads', type=int, default=1)
    parser.add_argument('--t_layers', type=int, default=1)
    parser.add_argument('--t_dropout', type=float, default=0.2)
    parser.add_argument('--t_dim', type=int, default=-1)

    # optimization parameters
    parser.add_argument('--orig_imsize', type=int,
                        default=-1)  # -1 for no cache, and -2 for no resize, only for MiniImageNet and CUB
    parser.add_argument('--train_size', type=int,
                        default=84)
    parser.add_argument('--test_size', type=int,
                        default=84)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_mul', type=float, default=10)
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine', 'constant'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.5)

    parser.add_argument('--augment', type=str, default='none')
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--m', type=int, default=2)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--init_weights', type=str, default=None)
    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # we find this weight decay value works the best
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--ps', type=str, default='')
    return parser


writer = None


def init_summary_writer(name):
    global writer
    writer = SummaryWriter(log_dir='runs/%s' % name)


def get_summary_writer():
    global writer
    return writer


def get_dataset(dataset, setname, unsupervised, args, augment='none'):
    Dataset = dataset_dict[dataset]
    return Dataset(setname, unsupervised, args, augment)
