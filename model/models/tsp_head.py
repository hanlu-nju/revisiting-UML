import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # (num_tasks, num_support, num_way, num_emb)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class TSPHead(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        hdim = self.hdim
        t_dim = args.t_dim
        if t_dim <= 0:
            t_dim = hdim
        layers = (MultiHeadAttention(args.t_heads, hdim, t_dim, t_dim, dropout=args.t_dropout) for _ in
                  range(args.t_layers))
        self.slf_attn = torch.nn.ModuleList(layers)

    def _forward(self, instance_embs, support_idx, query_idx, **kwargs):
        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(*(query_idx.shape + (-1,)))
        # if self.training:
        #     query, support = self.forward_head(query, support)
        #     logits = self.compute_logits(query, support)
        #     return logits, None
        # else:
        #     if self.args.with_head:
        #         query, support = self.forward_head(query, support)
        #     logits = self.compute_logits(query, support)
        #     return logits

        if self.training:
            query, support = self.forward_head(query, support)
            logits = self.compute_logits(query, support)
            return logits, None
        else:
            logits = self.compute_logits(query, support)
            return logits

        # for regularization
        # if self.training:
        #     aux_task = torch.cat([support.view(self.args.num_tasks, self.args.shot, self.args.way, emb_dim),
        #                           query.view(self.args.num_tasks, self.args.query, self.args.way, emb_dim)],
        #                          1)  # T x (K+Kq) x N x d
        #     num_query = np.prod(aux_task.shape[1:3])
        #     aux_task = aux_task.permute([0, 2, 1, 3])
        #     aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
        #     # apply the transformation over the Aug Task
        #     aux_emb = self.slf_attn(aux_task, aux_task, aux_task)  # T x N x (K+Kq) x d
        #     # compute class mean
        #     aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
        #     aux_center = torch.mean(aux_emb, 2)  # T x N x d
        #
        #     if self.args.use_euclidean:
        #         aux_task = aux_task.permute([1, 0, 2]).contiguous().view(-1, emb_dim).unsqueeze(
        #             1)  # (Nbatch*Nq*Nw, 1, d)
        #         aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        #         aux_center = aux_center.view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d)
        #
        #         logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
        #     else:
        #         aux_center = F.normalize(aux_center, dim=-1)  # normalize for cosine distance
        #         aux_task = aux_task.permute([1, 0, 2]).contiguous().view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)
        #
        #         logits_reg = torch.bmm(aux_task, aux_center.permute([0, 2, 1])) / self.args.temperature2
        #         logits_reg = logits_reg.view(-1, num_proto)
        #     label_aux = torch.arange(args.way, dtype=torch.long).repeat(
        #         args.num_tasks * (args.shot + args.query)  # *(self.train_loader.num_device if args.multi_gpu else 1)
        #     ).to(args.device)
        #     loss_reg = F.cross_entropy(logits_reg, label_aux)
        #     return logits, loss_reg
        # else:
        #     return logits

    def compute_logits(self, query, support):
        '''
        :param proto: (num_task,num_way,emb_dim)
        :param query: (num_task,num_query,num_way,emb_dim)
        :param max_pool:
        :return:
        '''
        proto = support.mean(dim=1)  # Ntask x NK x d
        emb_dim = proto.size(-1)
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query.shape[1:3])
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if self.args.similarity == 'euclidean':
            query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else:
            if self.args.similarity == 'sns':
                proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            elif self.args.similarity == 'cosine':
                proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
                query = F.normalize(query, dim=-1)
            else:
                assert self.args.similarity == 'dot'
                # proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.args.temperature
            logits = logits.view(-1, num_proto)
        return logits

    def forward_head(self, query, support):
        args = self.args
        s, q = support.size(1), query.size(1)
        all = torch.cat((support, query), dim=1)
        original_shape = all.shape
        all = all.view(all.shape[0], -1, all.shape[-1])
        for attn in self.slf_attn:
            all = attn(all, all, all)
        support, query = torch.split(all.view(*original_shape), (s, q), dim=1)
        return query, support
