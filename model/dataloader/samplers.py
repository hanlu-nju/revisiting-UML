import torch
import numpy as np



class CategoriesSampler:

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # take n way from l labels
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]  # take k+q examples
                batch.append(l[pos])
            # (way, shot + query) -> (shot + query, way)
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class RandomSampler:

    def __init__(self, label, n_batch, n_per):
        self.n_batch = n_batch
        self.n_per = n_per
        self.label = np.array(label)
        self.num_label = self.label.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = torch.randperm(self.num_label)[:self.n_per]
            yield batch


class RandomExclusiveSampler:
    def __init__(self, label, n_batch, n_per):
        self.n_per = n_per
        self.label = np.array(label)
        self.num_label = self.label.shape[0]
        if n_batch <= 0:
            self.n_batch = self.num_label // self.n_per
        else:
            self.n_batch = n_batch

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        randperm = torch.randperm(self.num_label)
        chunks = torch.split(randperm, self.n_per)
        for i_batch in range(self.n_batch):
            batch = chunks[i_batch]
            yield batch


# sample for each class
class ClassSampler:

    def __init__(self, label, n_per=None):
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return len(self.m_ind)

    def __iter__(self):
        classes = torch.arange(len(self.m_ind))
        for c in classes:
            l = self.m_ind[int(c)]
            if self.n_per is None:
                pos = torch.randperm(len(l))
            else:
                pos = torch.randperm(len(l))[:self.n_per]
            yield l[pos]


# for ResNet Fine-Tune, which output the same index of task examples several times
class InSetSampler:

    def __init__(self, n_batch, n_sbatch, pool):  # pool is a tensor
        self.n_batch = n_batch
        self.n_sbatch = n_sbatch
        self.pool = pool
        self.pool_size = pool.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = self.pool[torch.randperm(self.pool_size)[:self.n_sbatch]]
            yield batch


class NegativeSampler(CategoriesSampler):

    def __init__(self, args, label, n_batch, n_cls, n_per):
        super().__init__(label, n_batch, n_cls, n_per)
        self.args = args
        self.total_bidx = torch.ones(len(label)).bool()
        self.total_iidx = torch.arange(len(label))

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # take n way from l labels
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]  # take k+q examples
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            tmp_bidx = self.total_bidx.clone()
            tmp_bidx[batch] = False
            neg_idx = torch.from_numpy(np.random.choice(self.total_iidx[tmp_bidx], self.args.num_negative))
            batch = torch.cat([batch, neg_idx])
            yield batch
