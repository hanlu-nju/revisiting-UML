from abc import ABC

import torch
import torch.nn as nn

from model.networks import res18, resnet
from model.utils import get_summary_writer, Averager


def record_grad_norm(module):
    def hook(grad):
        norms = torch.norm(grad, dim=1)
        norm = torch.sum(norms)
        module.grad_norm_accumulator.add(norm.item(), norms.size(0))

    return hook


class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.construct_encoder(args)
        # self.encoder = HeadWrapper(self.encoder, self.hdim, args)

        self.ep = 0
        self.gep = 0
        self.lep = 0
        self.writer = get_summary_writer()
        self.grad_norm_accumulator = Averager()
        self.emb_norm_accumulator = Averager()

    def construct_encoder(self, args):
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import convnet
            self.hdim = 64
            self.encoder = convnet(True)
        elif args.backbone_class == 'ConvNetF':
            from model.networks.convnet import convnet
            self.hdim = 1600
            self.encoder = convnet(False)
        elif args.backbone_class == 'Res12':
            self.hdim = 640
            from model.networks.res12 import ResNet
            params = {}
            if args.dataset in ['CIFAR-FS', 'FC100']:
                params['drop_block'] = False
            self.encoder = ResNet(**params)
        elif args.backbone_class in res18.__all__:
            # from model.networks.res18 import ResNet
            self.encoder = getattr(res18, args.backbone_class)()
            self.hdim = 512 * self.encoder.expansion
        elif args.backbone_class in resnet.__all__:
            self.hdim = 64
            # from model.networks.res18 import ResNet
            self.encoder = getattr(resnet, args.backbone_class)()
        elif args.backbone_class == 'WRN':
            self.hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10,
                                       0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        else:
            raise ValueError('Unrecognized network structure')

    def split_instances_normal(self, num_tasks, num_shot, num_query, num_way, num_class=None):
        num_class = num_way if (num_class is None or num_class < num_way) else num_class

        permuted_ids = torch.zeros(num_tasks, num_shot + num_query, num_way).long()
        for i in range(num_tasks):
            # select class indices
            clsmap = torch.randperm(num_class)[:num_way]
            # ger permuted indices
            for j, clsid in enumerate(clsmap):
                permuted_ids[i, :, j].copy_(
                    torch.randperm((num_shot + num_query)) * num_class + clsid
                )

        if torch.cuda.is_available():
            permuted_ids = permuted_ids.cuda()

        support_idx, query_idx = torch.split(permuted_ids, [num_shot, num_query], dim=1)
        return support_idx, query_idx

    def split_instances(self, data):
        args = self.args
        if self.training:
            if args.unsupervised:
                return self.split_instances_normal(args.num_tasks, args.shot,
                                                   args.query, args.way, args.batch_size)
            return self.split_instances_normal(args.num_tasks, args.shot,
                                               args.query, args.way, args.num_classes)
        else:
            return self.split_instances_normal(1, args.eval_shot,
                                               args.eval_query, args.eval_way)

    def forward(self, x, get_feature=False, **kwargs):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            if self.training:
                logits, loss_reg = self._forward(instance_embs, support_idx, query_idx, **kwargs)
                instance_embs.register_hook(record_grad_norm(self))
                norms = torch.norm(instance_embs, dim=1)
                norm = torch.sum(norms)
                self.emb_norm_accumulator.add(norm.item(), norms.size(0))
                label = torch.arange(self.args.way, dtype=torch.long).repeat(
                    self.args.num_tasks * self.args.query  # *(self.train_loader.num_device if args.multi_gpu else 1)
                ).to(self.args.device)
                return logits, loss_reg, label
            else:
                logits = self._forward(instance_embs, support_idx, query_idx, **kwargs)
                return logits

    def _forward(self, instance_embs, support_idx, query_idx, **kwargs):
        # emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
        return self._forward_task(support, query, **kwargs)

    def _forward_task(self, support, query, **kwargs):
        pass

    def set_epoch(self, ep):
        self.gep = ep
        self.lep = 0
        self.clear_statistics()

    def clear_statistics(self):
        self.grad_norm_accumulator.reset()
        self.emb_norm_accumulator.reset()

    def statistics(self):
        return {'emb_grad_norm': self.grad_norm_accumulator.item(),
                'emb_norm': self.emb_norm_accumulator.item()}

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     return self.encoder.state_dict(destination, prefix, keep_vars)
    #
    # def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
    #                     strict: bool = True):
    #     return self.encoder.load_state_dict(state_dict, strict)

    # def prepare_label(self):
    #     args = self.args
    #
    #     # prepare one-hot label
    #
    #     # label_aux = torch.arange(args.way, dtype=torch.int8).repeat(
    #     #     args.num_tasks * (args.shot + args.query)  # *(self.train_loader.num_device if args.multi_gpu else 1)
    #     # ).to(args.device)
    #
    #     return label


class FewShotModelWrapper(FewShotModel, ABC):
    def __init__(self, args, model: FewShotModel):
        super().__init__(args)
        self.model = model
        self.encoder = self.model.encoder

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination, prefix, keep_vars)
