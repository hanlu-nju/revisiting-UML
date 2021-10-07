import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.dataloader.samplers import CategoriesSampler
from model.models import wrappers
from model.models.protonet import ProtoNet
from model.models.tsp_head import TSPHead
from model.optimizer.lars import LARS
from model.utils import get_dataset


class MultiGPUDataloader:
    def __init__(self, dataloader, num_device):
        self.dataloader = dataloader
        self.num_device = num_device

    def __len__(self):
        return len(self.dataloader) // self.num_device

    def __iter__(self):
        data_iter = iter(self.dataloader)
        done = False

        while not done:
            try:
                output_batch = ([], [])
                for _ in range(self.num_device):
                    batch = next(data_iter)
                    for i, v in enumerate(batch):
                        output_batch[i].append(v[None])

                yield (torch.cat(_, dim=0) for _ in output_batch)
            except StopIteration:
                done = True
        return


def examplar_collate(batch):
    X, Y = [], []
    for b in batch:
        X.append(torch.stack(b[0]))
        Y.append(b[1])
    X = torch.stack(X)
    label = torch.LongTensor(Y)
    img = torch.cat(tuple(X.permute(1, 0, 2, 3, 4)), dim=0)
    # (repeat * class , *dim)
    return img, label


def get_dataloader(args):
    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch * num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers = args.num_workers * num_device if args.multi_gpu else args.num_workers

    trainset = get_dataset(args.dataset, 'train', args.unsupervised, args, augment=args.augment)

    args.num_classes = min(len(trainset.wnids), args.num_classes)

    if args.unsupervised:
        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=examplar_collate,
                                  pin_memory=True, drop_last=True)
    else:
        train_sampler = CategoriesSampler(trainset.label,
                                          num_episodes,
                                          max(args.way, args.num_classes),
                                          args.shot + args.query)

        train_loader = DataLoader(dataset=trainset,
                                  num_workers=num_workers,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)

    valset = get_dataset(args.dataset, 'val', args.unsupervised, args)
    testsets = dict(((n, get_dataset(n, 'test', args.unsupervised, args)) for n in args.eval_dataset.split(',')))
    args.image_shape = trainset.image_shape
    return train_loader, valset, testsets


def prepare_model(args):
    args.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = eval(args.model_class)(args)

    # load pre-trained model (no FC weights)
    if args.init_weights is not None:
        model_dict = model.state_dict()
        # if args.augment == 'moco':
        #     pretrained_dict = torch.load(args.init_weights)['state_dict']
        #     pretrained_dict = {'encoder' + k[len('encoder_q'):]: v for k, v in pretrained_dict.items() if
        #                        k.startswith('encoder_q')}
        # else:
        try:
            pretrained_dict = torch.load(args.init_weights, map_location=args.device)
        except:
            import pickle

            with open(args.init_weights, 'rb') as fp:
                pretrained_dict = pickle.load(fp)
        keys = ['params', 'state_dict']
        for k in keys:
            if k in pretrained_dict:
                pretrained_dict = pretrained_dict[k]
                break
        # pretrained_dict = torch.load(args.init_weights)['params']
        # if args.backbone_class == 'ConvNet':
        #     pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.additional != 'none':
        model = getattr(wrappers, args.additional + 'Wrapper')(args, model)
        # model = TaskContrastiveWrapper(args, model)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = model.to(device)
    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)
        para_model = model.to(device)
    else:
        para_model = model.to(device)
    # if args.finetune:
    #     model.eval()
    #     para_model.eval()
    # print(model.state_dict().keys())
    return model, para_model


def prepare_optimizer(model, args):
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]
    print('top params', [k for k, v in model.named_parameters() if 'encoder' not in k])
    # as in the literature, we use ADAM for ConvNet and SGD for other backbones
    param_groups = [{'params': model.encoder.parameters()},
                    {'params': top_para, 'lr': args.lr * args.lr_mul}]
    # param_groups = model.parameters()
    # param = dict(model.named_parameters())
    if args.lars:
        optimizer = LARS(param_groups,
                         lr=args.lr,
                         momentum=args.mom,
                         weight_decay=args.weight_decay)
    else:
        if args.backbone_class in ['ConvNet']:
            optimizer = optim.Adam(
                param_groups,
                lr=args.lr,
                # weight_decay=args.weight_decay, do not use weight_decay here
            )
        else:
            optimizer = optim.SGD(param_groups,
                                  lr=args.lr,
                                  momentum=args.mom,
                                  nesterov=True,
                                  weight_decay=args.weight_decay
                                  )

    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.step_size),
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(_) for _ in args.step_size.split(',')],
            gamma=args.gamma,
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.max_epoch,
            eta_min=0  # a tuning parameter
        )
    elif args.lr_scheduler == 'constant':
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda ep: 1  # a tuning parameter
        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler
