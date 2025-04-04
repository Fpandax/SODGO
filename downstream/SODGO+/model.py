from helper import *
import torch.nn as nn
from Layer import *

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size): # hidden_size:200 x_dim:200 y_dim:200
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        #         print(x_samples.size())
        #         print(y_samples.size())
        mu, logvar = self.get_mu_logvar(x_samples)

        return (-(mu - y_samples) ** 2 / 2. / logvar.exp()).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
         
        self.bceloss = torch.nn.BCEWithLogitsLoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)



class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)  
         
        weight = nn.Parameter(torch.from_numpy(weight)) #  (100,600)
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
         
        stdv = 1. / np.sqrt(self.weight.size(1)) # weight.size(1)：600
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    # fc
    def forward(self, x):
         
        x = x.to(self.weight.device)
        return torch.mm(x, self.weight) + self.bias


class CapsuleBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_ent, num_rel, num_namespaces, params=None
                 ,node_type=None, go_namespace=None, type2id=None, init_goemb=None):
        super(CapsuleBase, self).__init__(params)
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.device = self.edge_index.device
         
        # self.init_genenembed = get_param((num_ent, self.p.init_dim))
        # self.init_embed = get_goenembed(self.init_goemb, self.init_genenembed)
         
        if self.p.textovector:
            self.init_embed = initialize_embedding(self.init_goemb, (num_ent, self.p.init_dim))
        else:
            self.init_embed = get_param((num_ent, self.p.init_dim))
        self.init_rel = get_param((num_rel * 2, self.p.gcn_dim))
        self.pca = SparseInputLinear(self.p.init_dim, self.p.num_factors * self.p.gcn_dim)

        self.node_type = node_type
        self.go_namespace = go_namespace
        self.type2id = type2id
        self.init_goemb = init_goemb

        conv_ls = []
        for i in range(self.p.gcn_layer):
            conv = DisenLayer(self.edge_index, self.edge_type, self.p.init_dim, self.p.gcn_dim, num_rel, num_namespaces,
                              act=self.act, params=self.p, head_num=self.p.head_num,
                              node_type=self.node_type, go_namespace=self.go_namespace,
                              type2id=self.type2id)
            self.add_module('conv_{}'.format(i), conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls  # DisenLayer(100,200,num_rels=10)
        if self.p.mi_train:
            if self.p.mi_method == 'club_b':
                num_dis = int((self.p.num_factors) * (self.p.num_factors - 1) / 2)
                # print(num_dis)
                self.mi_Discs = nn.ModuleList(
                    [CLUBSample(self.p.gcn_dim, self.p.gcn_dim, self.p.gcn_dim) for fac in range(num_dis)])
            elif self.p.mi_method == 'club_s':
                self.mi_Discs = nn.ModuleList(
                    [CLUBSample((fac + 1) * self.p.gcn_dim, self.p.gcn_dim, (fac + 1) * self.p.gcn_dim) for fac in
                     range(self.p.num_factors - 1)])

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.rel_drop = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def lld_bst(self, sub, rel, drop1, mode='train'):
        x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim)  # [N K F]
        r = self.init_rel
        for conv in self.conv_ls:
            x, r = conv(x, r, mode)  # N K F
            if self.p.mi_drop:
                x = drop1(x)
            else:
                continue

        sub_emb = torch.index_select(x, 0, sub)
        lld_loss = 0.
        sub_emb = sub_emb.view(-1, self.p.gcn_dim * self.p.num_factors)
        if self.p.mi_method == 'club_s':
            for i in range(self.p.num_factors - 1):
                bnd = i + 1
                lld_loss += self.mi_Discs[i].learning_loss(sub_emb[:, :bnd * self.p.gcn_dim],
                                                           sub_emb[:, bnd * self.p.gcn_dim: (bnd + 1) * self.p.gcn_dim])

        elif self.p.mi_method == 'club_b':
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    lld_loss += self.mi_Discs[cnt].learning_loss(
                        sub_emb[:, i * self.p.gcn_dim: (i + 1) * self.p.gcn_dim],
                        sub_emb[:, j * self.p.gcn_dim: (j + 1) * self.p.gcn_dim])
                    cnt += 1
        return lld_loss

    def mi_cal(self, sub_emb):
        def loss_dependence_hisc(zdata_trn, ncaps, nhidden):
            loss_dep = torch.zeros(1).cuda()
            hH = (-1 / nhidden) * torch.ones(nhidden, nhidden).cuda() + torch.eye(nhidden).cuda()
            kfactor = torch.zeros(ncaps, nhidden, nhidden).cuda()

            for mm in range(ncaps):
                data_temp = zdata_trn[:, mm * nhidden:(mm + 1) * nhidden]
                kfactor[mm, :, :] = torch.mm(data_temp.t(), data_temp)

            for mm in range(ncaps):
                for mn in range(mm + 1, ncaps):
                    mat1 = torch.mm(hH, kfactor[mm, :, :])
                    mat2 = torch.mm(hH, kfactor[mn, :, :])
                    mat3 = torch.mm(mat1, mat2)
                    teststat = torch.trace(mat3)

                    loss_dep = loss_dep + teststat
            return loss_dep

        def loss_dependence_club_s(sub_emb):
            mi_loss = 0.
            for i in range(self.p.num_factors - 1):
                bnd = i + 1
                mi_loss += self.mi_Discs[i](sub_emb[:, :bnd * self.p.gcn_dim],
                                            sub_emb[:, bnd * self.p.gcn_dim: (bnd + 1) * self.p.gcn_dim])
            return mi_loss

        def loss_dependence_club_b(sub_emb):
            mi_loss = 0.
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    mi_loss += self.mi_Discs[cnt](sub_emb[:, i * self.p.gcn_dim: (i + 1) * self.p.gcn_dim],
                                                  sub_emb[:, j * self.p.gcn_dim: (j + 1) * self.p.gcn_dim])
                    cnt += 1
            return mi_loss

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        if self.p.mi_method == 'club_s':
            mi_loss = loss_dependence_club_s(sub_emb)
        elif self.p.mi_method == 'club_b':
            mi_loss = loss_dependence_club_b(sub_emb)
        elif self.p.mi_method == 'hisc':
            mi_loss = loss_dependence_hisc(sub_emb, self.p.num_factors, self.p.gcn_dim)
        elif self.p.mi_method == "dist":
            cor = 0.
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    cor += DistanceCorrelation(sub_emb[:, i * self.p.gcn_dim: (i + 1) * self.p.gcn_dim],
                                               sub_emb[:, j * self.p.gcn_dim: (j + 1) * self.p.gcn_dim])
            return cor
        else:
            raise NotImplementedError

        return mi_loss

    def forward_base(self, sub, rel, drop1, drop2, mode):
        if not self.p.no_enc:
             
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim)  # [42444 3 200]
            #init r_embed
            r = self.init_rel # (21,200)
            for conv in self.conv_ls:
                x, r = conv(x, r, mode)  # N K F
                x = drop1(x)
        else:
            x = self.init_embed
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)
        mi_loss = 0.
        sub_emb = sub_emb.view(-1, self.p.gcn_dim * self.p.num_factors)
        mi_loss = self.mi_cal(sub_emb)

        return sub_emb, rel_emb, x, mi_loss

    def test_base(self, sub, rel, drop1, drop2, mode):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim)  # [N K F]
            r = self.init_rel
            for conv in self.conv_ls:
                x, r = conv(x, r, mode)  # N K F
                x = drop1(x)
        else:
            x = self.init_embed.view(-1, self.p.num_factors, self.p.gcn_dim)
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)

        return sub_emb, rel_emb, x, 0.

class SODGO_InteractE(CapsuleBase):
    def __init__(self, edge_index, edge_type, params=None
                 , node_type=None, go_namespace=None, type2id=None, init_goemb=None):
        self.node_type = node_type
        self.go_namespace = go_namespace
        self.type2id = type2id
        self.init_goemb = init_goemb

        super(self.__class__, self).__init__(edge_index, edge_type,params.num_ent, params.num_rel, params.num_namespace,
                                             params, self.node_type, self.go_namespace, self.type2id, self.init_goemb)

        self.inp_drop = torch.nn.Dropout(self.p.iinp_drop)

        self.feature_map_drop = torch.nn.Dropout2d(self.p.ifeat_drop)
        self.hidden_drop = torch.nn.Dropout(self.p.ihid_drop)

        self.hidden_drop_gcn = torch.nn.Dropout(0)

        self.bn0 = torch.nn.BatchNorm2d(self.p.iperm)
        flat_sz_h = self.p.ik_h
        flat_sz_w = 2 * self.p.ik_w
        self.padding = 0

        self.bn1 = torch.nn.BatchNorm2d(self.p.inum_filt * self.p.iperm)
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.inum_filt * self.p.iperm

        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        self.chequer_perm = self.get_chequer_perm()
        if self.p.score_method.startswith('cat'):
            self.fc_a = nn.Linear(2 * self.p.gcn_dim, 1)
        elif self.p.score_method == 'learn':
            self.fc_att = get_param((2 * self.p.num_rel, self.p.num_factors))
        self.rel_weight = self.conv_ls[-1].rel_weight
        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.register_parameter('conv_filt',
                                Parameter(torch.zeros(self.p.inum_filt, 1, self.p.iker_sz, self.p.iker_sz)))
        xavier_normal_(self.conv_filt)

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.inp_drop)

    def forward(self, sub, rel, neg_ents=None, mode='train'):
        if mode == 'train' and self.p.mi_train:
            sub_emb, rel_emb, all_ent, corr = self.forward_base(sub, rel, self.inp_drop, self.hidden_drop_gcn, mode)
            # sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        else:
            sub_emb, rel_emb, all_ent, corr = self.test_base(sub, rel, self.inp_drop, self.hidden_drop_gcn, mode)
        sub_emb = sub_emb.view(-1, self.p.gcn_dim)
        rel_emb = rel_emb.view(-1, self.p.gcn_dim)
        all_ent = all_ent.view(-1, self.p.num_factors, self.p.gcn_dim)
        # sub: [B K F]  
        # rel: [B K F] 
        # all_ent: [N K F]
        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.p.iperm, 2 * self.p.ik_w, self.p.ik_h))
        stack_inp = self.bn0(stack_inp)
        x = stack_inp
        x = self.circular_padding_chw(x, self.p.iker_sz // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.p.iperm, 1, 1, 1), padding=self.padding, groups=self.p.iperm)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)  # [B*K F]
        x = x.view(-1, self.p.num_factors, self.p.gcn_dim)

        # start to calculate the attention
        rel_weight = torch.index_select(self.rel_weight, 0, rel)  # B K F
        rel_emb = rel_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)  # B K F
        if self.p.score_method == 'dot_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))  # B K
        elif self.p.score_method == 'dot_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))  # B K
        elif self.p.score_method == 'cat_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze())  # B K
        elif self.p.score_method == 'cat_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze())  # B K
        elif self.p.score_method == 'learn':
            att_rel = torch.index_select(self.fc_att, 0, rel)
            attention = self.leakyrelu(att_rel)  # [B K]
        attention = nn.Softmax(dim=-1)(attention)


        if self.p.strategy == 'one_to_n' or neg_ents is None:
            x = torch.einsum('bkf,nkf->bkn', [x, all_ent])
            x += self.bias.expand_as(x)
        else:
            x = torch.mul(x.unsqueeze(1), all_ent[neg_ents]).sum(dim=-1)
            x += self.bias[neg_ents].expand_as(x)

        if self.p.score_order == 'before':
            x = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.sigmoid(x)
        elif self.p.score_order == 'after':
            x = torch.sigmoid(x)
            if self.p.strategy == 'one_to_n' or neg_ents is None:
                pred = torch.einsum('bk,bkn->bn', [attention, x])
            else:
                pred = torch.einsum('bk,bnk->bn', [attention, x])
            pred = torch.clamp(pred, min=0., max=1.0)
        return pred, corr, all_ent

    def get_chequer_perm(self):
        """
        Function to generate the chequer permutation required for InteractE model

        Parameters
        ----------

        Returns
        -------

        """
        ent_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])  # [1,200]
        rel_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])

        comb_idx = []
        for k in range(self.p.iperm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.p.ik_h):
                for j in range(self.p.ik_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm
