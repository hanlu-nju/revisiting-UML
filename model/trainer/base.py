import abc
import os.path as osp

import torch
from torch.utils.data import DataLoader

from model.dataloader.samplers import CategoriesSampler, NegativeSampler
from model.logger import Logger
from model.utils import Averager, Timer
from model.utils import init_summary_writer, get_summary_writer
import os


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        if args.dataset == 'CUB':
            self.VAL_SETTING = [(5, 1), (5, 5), (5, 20)]
        else:
            self.VAL_SETTING = [(5, 1), (5, 5), (5, 20), (5, 50)]
        if args.eval_dataset == 'CUB':
            self.TEST_SETTINGS = [(5, 1), (5, 5), (5, 20)]
        else:
            self.TEST_SETTINGS = [(5, 1), (5, 5), (5, 20), (5, 50)]
        self.args = args
        # ensure_path(
        #     self.args.save_path,
        #     scripts_to_save=['model/models', 'model/networks', __file__],
        # )
        self.logger = Logger(args, osp.join(args.save_path))
        init_summary_writer(args.filename)
        self.writer = get_summary_writer()
        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc_interval'] = 0.0

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, data_loader):
        pass

    @abc.abstractmethod
    def evaluate_test(self, data_loader):
        pass

    @abc.abstractmethod
    def final_record(self):
        pass

    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch % args.eval_interval == 0:
            # if args.eval_all:
            #     for i, (args.eval_way, args.eval_shot) in enumerate(self.VAL_SETTING):
            #         if i == 0:
            #             vl, va, vap = self.eval_process(args, epoch)
            #         else:
            #             self.eval_process(args, epoch)
            # else:
            vl, va, vap = self.eval_process(args, epoch)
            if va >= self.trlog['max_acc']:
                self.trlog['max_acc'] = va
                self.trlog['max_acc_interval'] = vap
                self.trlog['max_acc_epoch'] = self.train_epoch
                self.save_model('max_acc')
            print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))

    def eval_process(self, args, epoch):
        valset = self.valset
        val_sampler = CategoriesSampler(valset.label,
                                        args.num_eval_episodes,
                                        args.eval_way, args.eval_shot + args.eval_query)
        val_loader = DataLoader(dataset=valset,
                                batch_sampler=val_sampler,
                                num_workers=args.num_workers,
                                pin_memory=True)
        vl, va, vap = self.evaluate(val_loader)
        self.logger.add_scalar('%dw%ds_val_loss' % (args.eval_way, args.eval_shot), float(vl),
                               self.train_epoch)
        self.logger.add_scalar('%dw%ds_val_acc' % (args.eval_way, args.eval_shot), float(va),
                               self.train_epoch)
        self.writer.add_scalar('val loss', vl, epoch)
        self.writer.add_scalar('val acc', va, epoch)
        print('epoch {},{} way {} shot, val, loss={:.4f} acc={:.4f}+{:.4f}'.format(epoch, args.eval_way,
                                                                                   args.eval_shot, vl, va,
                                                                                   vap))
        return vl, va, vap

    def try_logging(self, tl1, tl2, ta, tg=None):
        args = self.args
        if self.train_step % args.log_interval == 0:
            print('epoch {}, train {:06g}/{:06g}, total loss={:.4f}, loss={:.4f} acc={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.train_step,
                          self.max_steps,
                          tl1.item(), tl2.item(), ta.item(),
                          self.optimizer.param_groups[0]['lr']))
            self.logger.add_scalar('train_total_loss', tl1.item(), self.train_step)
            self.logger.add_scalar('train_loss', tl2.item(), self.train_step)
            self.logger.add_scalar('train_acc', ta.item(), self.train_step)
            if tg is not None:
                self.logger.add_scalar('grad_norm', tg.item(), self.train_step)
            print('data_timer: {:.2f} sec, ' \
                  'forward_timer: {:.2f} sec,' \
                  'backward_timer: {:.2f} sec, ' \
                  'optim_timer: {:.2f} sec'.format(
                self.dt.item(), self.ft.item(),
                self.bt.item(), self.ot.item())
            )
            self.logger.dump()

    def save_model(self, name, symlink=False, link_path=None):
        if symlink:
            assert link_path is not None
            linkname = osp.join(self.args.save_path, f'{name}.pth')
            if os.path.islink(linkname):
                os.remove(linkname)
            os.symlink(link_path, linkname)
        else:
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, name + '.pth')
            )

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
