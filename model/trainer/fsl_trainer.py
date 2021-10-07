import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataloader.samplers import CategoriesSampler
from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    Averager, count_acc, compute_confidence_interval,
)
from .helpers import examplar_collate
import os


# inter_grad = {}
#
#
# def save_grad(name):
#     def hook(grad):
#         inter_grad[name] = grad
#
#     return hook


class FSLTrainer(Trainer):
    def final_record(self):
        pass

    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.valset, self.testset = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)

        # for n, p in self.para_model.named_parameters():
        #     p.register_hook(save_grad(n))
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        self.stats = {}

    # @profile
    def train(self):
        args = self.args
        # self.model.train()

        # start FSL training
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            # if self.args.fix_running:
            #     self.model.encoder.eval()
            self.model.set_epoch(epoch)
            tl1 = Averager()
            ta = Averager()

            start_tm = time.time()
            pbar = tqdm(total=len(self.train_loader), desc='epoch %d' % epoch)
            for batch in self.train_loader:
                self.train_step += 1

                self.optimizer.zero_grad()
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data, _ = batch[0], batch[1]

                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                logits, reg_loss, label = self.para_model(data)

                loss = F.cross_entropy(logits, label)
                if reg_loss is None:
                    reg_loss = 0
                total_loss = loss + args.balance * reg_loss

                # tl2.add(loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)

                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)

                # refresh start_tm
                start_tm = time.time()
                pbar.set_description(f'epoch {epoch}, loss: {tl1.item():.4f}, acc: {ta.item():.4f}')
                pbar.update()
            pbar.close()
            for k, v in self.model.statistics().items():
                self.writer.add_scalar(k, v, epoch)
                if k not in self.stats:
                    self.stats[k] = []
                self.stats[k].append(v)
            self.writer.add_scalar('train acc', ta.item(), epoch)
            self.writer.add_scalar('train loss', tl1.item(), epoch)
            self.lr_scheduler.step()
            self.try_evaluate(epoch)

            print('ETA:{}/{}'.format(
                self.timer.measure(),
                self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2))  # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(
            # args.num_tasks *
            args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader, 1), total=len(data_loader), desc='eval procedure'):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc

        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])

        # train mode
        self.model.train()
        if self.args.fix_running:
            self.model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self, **kwargs):
        # restore model args
        args = self.args
        # evaluation mode
        if args.unsupervised:
            models_pths = ['epoch-last.pth']
        else:
            models_pths = [
                'epoch-last.pth'
                , 'max_acc.pth'
            ]
        with open(osp.join(self.args.save_path,
                           'test_result%s' % ('_eval_all' if args.eval_all else '')),
                  'w') as f:
            for pth in models_pths:
                ensemble_result = []
                print('----------- test pth {} --------------'.format(pth))
                f.write('----------- test pth {} --------------\n'.format(pth))
                path = osp.join(self.args.save_path, pth)
                print('model path %s' % path)
                model_dict = self.model.state_dict()
                pretrained_dict = torch.load(path)
                keys = ['params', 'state_dict']
                for k in keys:
                    if k in pretrained_dict:
                        pretrained_dict = pretrained_dict[k]
                        break
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                print(pretrained_dict.keys())
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
                self.model.eval()
                print('best epoch {}, best val acc={:.2f} + {:.2f}'.format(
                    self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
                with torch.no_grad():
                    for d, testset in self.testset.items():
                        print('----------- test on {} --------------'.format(d))
                        f.write('----------- test on {} --------------\n'.format(d))
                        if args.eval_all:
                            for args.eval_way, args.eval_shot in testset.eval_setting:
                                vl, va, vap = self.test_process(testset)
                                f.write(
                                    '{} way {} shot,Test acc={:.2f} + {:.2f}\n'.format(args.eval_way, args.eval_shot,
                                                                                       va,
                                                                                       vap))
                                ensemble_result.append('{:.2f} + {:.2f}'.format(va, vap))
                        else:
                            vl, va, vap = self.test_process(testset)
                            f.write('{} way {} shot,Test acc={:.2f} + {:.2f}\n'.format(args.eval_way, args.eval_shot,
                                                                                       va,
                                                                                       vap))
                            ensemble_result.append('{:.2f} + {:.2f}'.format(va, vap))
                print('ensemble result: {}'.format(','.join(ensemble_result)))
                f.write('ensemble result: {}\n'.format(','.join(ensemble_result)))

    def evaluate_model(self, **kwargs):
        # restore model args
        args = self.args
        # evaluation mode
        path = kwargs['path']
        print('model path %s' % path)
        model_dict = self.model.state_dict()
        try:
            pretrained_dict = torch.load(path)
        except:
            import pickle
            with open(path, 'rb') as fp:
                pretrained_dict = pickle.load(fp)
        keys = ['params', 'state_dict']
        for k in keys:
            if k in pretrained_dict:
                pretrained_dict = pretrained_dict[k]
                break
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        with torch.no_grad():
            with open(osp.join(self.args.save_path,
                               'test_result%s' % ('_eval_all' if args.eval_all else '')),
                      'w') as f:
                self.test_on_datasets(f)

    def test_on_datasets(self, f, unsupervised=False):
        args = self.args
        ensemble_result = []
        for d, testset in self.testset.items():
            print('----------- {} test on {} --------------'.format('unsupervised' if unsupervised else '', d))
            f.write('----------- {} test on {} --------------'.format('unsupervised' if unsupervised else '', d))
            if args.eval_all:
                for args.eval_way, args.eval_shot in testset.eval_setting:
                    vl, va, vap = self.test_process(testset)
                    f.write('{} way {} shot,Test acc={:.2f} + {:.2f}\n'.format(args.eval_way, args.eval_shot,
                                                                               va,
                                                                               vap))
                    ensemble_result.append('{:.2f} + {:.2f}'.format(va, vap))
            else:
                vl, va, vap = self.test_process(testset)
                f.write('{} way {} shot,Test acc={:.2f} + {:.2f}\n'.format(args.eval_way, args.eval_shot,
                                                                           va,
                                                                           vap))
                ensemble_result.append('{:.2f} + {:.2f}'.format(va, vap))
        print('ensemble result: {}'.format(','.join(ensemble_result)))
        f.write('ensemble result: {}\n'.format(','.join(ensemble_result)))

    def test_process(self, testset):
        args = self.args
        test_sampler = CategoriesSampler(testset.label,
                                         args.num_test_episodes,  # args.num_eval_episodes,
                                         args.eval_way, args.eval_shot + args.eval_query)
        test_loader = DataLoader(dataset=testset,
                                 batch_sampler=test_sampler,
                                 num_workers=args.num_workers,
                                 pin_memory=True)
        record = np.zeros((args.num_test_episodes, 2))  # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.long).repeat(
            args.eval_query).to(self.args.device)

        for i, batch in tqdm(enumerate(test_loader, 1), total=len(test_loader)):
            data = batch[0]
            data = data.to(self.args.device)
            logits = self.model(data)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            record[i - 1, 0] = loss.item()
            record[i - 1, 1] = acc

        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])
        print('{} way {} shot,Test acc={:.2f} + {:.2f}\n'.format(args.eval_way, args.eval_shot,
                                                                 va,
                                                                 vap))

        return vl, va, vap
