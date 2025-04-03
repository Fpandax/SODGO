import torch.nn as nn
from DisenLayer import *
import os
from helper import *

class CLUBSample(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
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
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / 2. / logvar.exp()).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        sample_size = x_samples.shape[0]
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
        # 替换 BCELoss 为 BCEWithLogitsLoss
        self.bceloss = torch.nn.BCEWithLogitsLoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # 获取输入 x 的设备
        x_device = x.device

        # 将 weight 和 bias 移动到 x 的设备，而不改变它们的 nn.Parameter 类型
        self.weight.data = self.weight.data.to(x_device)
        self.bias.data = self.bias.data.to(x_device)

        # 执行矩阵乘法
        output = torch.mm(x, self.weight) + self.bias

        return output

class CapsuleBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None, node_attributes=None
                 , rel_strength=None, init_goemb=None):
        assert node_attributes is not None
        super(CapsuleBase, self).__init__(params)
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.device = self.edge_index.device
        self.node_attributes = node_attributes
        self.rel_strength = rel_strength
        # self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        if init_goemb is not None:
            #将字典init_goemb转换为tensor
            # Parse the string values into lists of floats
            init_goemb_array = np.array([np.fromstring(value.strip('[]'), sep=',') for value in init_goemb.values()])
            self.init_goemb = torch.from_numpy(init_goemb_array).float()
            if self.p.direct_init:
                self.init_embed = self.init_goemb
            else:
                reduced_x = dim_reduction(self.init_goemb, 128)
                self.init_embed = reduced_x.repeat(1, 3)
        else:
            self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.init_rel = get_param((num_rel * 2, self.p.gcn_dim))
        self.pca = SparseInputLinear(self.p.init_dim, self.p.num_factors * self.p.gcn_dim)
        conv_ls = []
        for i in range(self.p.gcn_layer):
            conv = DisenLayer(self.p.num_ent, self.edge_index, self.edge_type, self.p.init_dim, self.p.gcn_dim, num_rel,
                              act=self.act, params=self.p, node_attributes=self.node_attributes, rel_strength=self.rel_strength)
            self.add_module('conv_{}'.format(i), conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls

        if self.p.mi_train:
            num_dis = int((self.p.num_factors) * (self.p.num_factors - 1) / 2)
            self.mi_Discs = nn.ModuleList([CLUBSample(self.p.gcn_dim, self.p.gcn_dim, self.p.gcn_dim) for fac in range(num_dis)])

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.rel_drop = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def lld_bst(self, sub, rel, drop1, mode='train'):
        x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim)
        r = self.init_rel
        for conv in self.conv_ls:
            x, r, reg_loss = conv(x, r, mode)
            if self.p.mi_drop:
                x = drop1(x)
            else:
                continue
        sub_emb = torch.index_select(x, 0, sub)
        lld_loss = 0.
        sub_emb = sub_emb.view(-1, self.p.gcn_dim * self.p.num_factors)
        cnt = 0
        for i in range(self.p.num_factors):
            for j in range(i + 1, self.p.num_factors):
                lld_loss += self.mi_Discs[cnt].learning_loss(
                    sub_emb[:, i * self.p.gcn_dim: (i + 1) * self.p.gcn_dim],
                    sub_emb[:, j * self.p.gcn_dim: (j + 1) * self.p.gcn_dim])
                cnt += 1
        return lld_loss

    def mi_cal(self, sub_emb):
        def loss_dependence_club_b(sub_emb):
            mi_loss = 0.
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    mi_loss += self.mi_Discs[cnt](sub_emb[:, i * self.p.gcn_dim: (i + 1) * self.p.gcn_dim],
                                                  sub_emb[:, j * self.p.gcn_dim: (j + 1) * self.p.gcn_dim])
                    cnt += 1
            return mi_loss
        mi_loss = loss_dependence_club_b(sub_emb)
        return mi_loss

    def forward_base(self, sub, rel, obj, drop1, drop2, mode):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim)
            r = self.init_rel
            for conv in self.conv_ls:
                x, r, analy_alpha = conv(x, r, mode)
                x = drop1(x)
        else:
            x = self.init_embed
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel).repeat(1, self.p.num_factors)

        if obj is not None:
            obj_emb = torch.index_select(x, 0, obj).repeat(1, self.p.num_factors)
            sub_emb = sub_emb.view(-1, self.p.gcn_dim * self.p.num_factors)
            obj_emb = obj_emb.view(-1, self.p.gcn_dim * self.p.num_factors)
            mi_loss = self.mi_cal(sub_emb)
            return sub_emb, rel_emb, obj_emb, x, mi_loss
        else:
            sub_emb = sub_emb.view(-1, self.p.gcn_dim * self.p.num_factors)
            mi_loss = self.mi_cal(sub_emb)
            return sub_emb, rel_emb, x, mi_loss, analy_alpha

    def test_base(self, sub, rel, obj, drop1, drop2, mode):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim)
            r = self.init_rel
            for conv in self.conv_ls:
                x, r, analy_alpha = conv(x, r, mode)
                x = drop1(x)
        else:
            x = self.init_embed.view(-1, self.p.num_factors, self.p.gcn_dim)
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel).repeat(1, self.p.num_factors)
        return sub_emb, rel_emb, x, 0., 0.

class SODGO_TransE(CapsuleBase):
    def __init__(self, edge_index, edge_type, params=None, node_attributes=None, rel_strength=None, init_goemb=None):
        self.node_attributes = node_attributes
        self.rel_strength = rel_strength
        self.init_goemb = init_goemb
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params, node_attributes, self.rel_strength, self.init_goemb)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        self.rel_weight = self.conv_ls[-1].rel_weight
        gamma_init = torch.FloatTensor([self.p.init_gamma])
        if not self.p.fix_gamma:
            self.register_parameter('gamma', Parameter(gamma_init))

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.drop)

    def forward(self, sub,  rel, obj, neg_ents=None, mode='train'):
        if mode == 'train' and self.p.mi_train:
            sub_emb, rel_emb, all_ent, corr, analy_alpha = self.forward_base(sub, rel, None, self.drop, self.drop, mode)
            sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        else:
            sub_emb, rel_emb, all_ent, corr, analy_alpha = self.test_base(sub, rel, None, self.drop, self.drop, mode)
        rel_weight = torch.index_select(self.rel_weight, 0, rel)
        rel_emb = rel_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        sub_rel_emb = sub_emb * rel_weight
        rel_emb = rel_emb
        attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))
        attention = nn.Softmax(dim=-1)(attention)
        obj_emb = sub_emb + rel_emb
        x2 = torch.sum(obj_emb * obj_emb, dim=-1)
        y2 = torch.sum(all_ent * all_ent, dim=-1)
        xy = torch.einsum('bkf,nkf->bkn', [obj_emb, all_ent])
        x = self.gamma - (x2.unsqueeze(2) + y2.t() - 2 * xy)

        if self.p.strategy == 'one_to_n' or neg_ents is None:
            x = torch.einsum('bkf,fkn->bkf', [x, all_ent])
            x += self.bias.expand_as(x)
        else:
            selected_all_ent = all_ent[neg_ents]
            selected_x = x.gather(2, neg_ents.unsqueeze(1).expand(-1, 3, -1))
            selected_all_ent = selected_all_ent.permute(0, 2, 1, 3)
            result = torch.mul(selected_x.unsqueeze(-1), selected_all_ent).sum(dim=-1)
            expanded_bias = self.bias[neg_ents].unsqueeze(1).expand(-1, 3, -1)
            result += expanded_bias
            x = result

        if self.p.score_order == 'before':
            x = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.sigmoid(x)
        elif self.p.score_order == 'after':
            x = torch.sigmoid(x)
            pred = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.clamp(pred, min=0., max=1.0)
        return pred, corr, all_ent, analy_alpha
class SODGO_InteractE(CapsuleBase):
    def __init__(self, edge_index, edge_type, params=None, node_attributes=None, rel_strength=None, init_goemb=None):
        self.node_attributes = node_attributes
        self.rel_strength = rel_strength
        self.init_goemb = init_goemb
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params, node_attributes,
                                             self.rel_strength, self.init_goemb)
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
        else:
            sub_emb, rel_emb, all_ent, corr = self.test_base(sub, rel, self.inp_drop, self.hidden_drop_gcn, mode)
        sub_emb = sub_emb.view(-1, self.p.gcn_dim)
        rel_emb = rel_emb.view(-1, self.p.gcn_dim)
        all_ent = all_ent.view(-1, self.p.num_factors, self.p.gcn_dim)
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
        x = F.relu(x)
        x = x.view(-1, self.p.num_factors, self.p.gcn_dim)
        rel_weight = torch.index_select(self.rel_weight, 0, rel)
        rel_emb = rel_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        if self.p.score_method == 'dot_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))
        elif self.p.score_method == 'dot_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))
        elif self.p.score_method == 'cat_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze())
        elif self.p.score_method == 'cat_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze())
        elif self.p.score_method == 'learn':
            att_rel = torch.index_select(self.fc_att, 0, rel)
            attention = self.leakyrelu(att_rel)
        attention = nn.Softmax(dim=-1)(attention)

        if self.p.strategy == 'one_to_n' or neg_ents is None:
            x = torch.einsum('bkf,nkf->bkn', [x, all_ent])
            x += self.bias.expand_as(x)
        else:
            x = torch.mul(x.unsqueeze(1), all_ent[neg_ents]).sum(dim=-1)
            x += self.bias[neg_ents]

        if self.p.score_order == 'before':
            x = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.sigmoid(x)
        elif self.p.score_order == 'after':
            x = torch.sigmoid(x)
            pred = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.clamp(pred, min=0., max=1.0)
        return pred, corr, all_ent

    def get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])
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


class DisenKGAT_Similarity(CapsuleBase):
    def __init__(self, edge_index, edge_type, params=None, node_attributes=None, rel_strength=None
                 , init_goemb=None):
        self.node_attributes = node_attributes
        self.rel_strength = rel_strength
        self.init_goemb = init_goemb
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params, node_attributes,
                                             self.rel_strength, self.init_goemb)


    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.drop)

    def forward(self, sub, rel, obj, neg_ents=None, mode='train'):
        if mode == 'train' and self.p.mi_train:
            sub_emb, rel_emb, obj_emb, all_ent, corr = self.forward_base(self, sub, rel, obj, self.drop, self.drop, mode)
            sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
            obj_emb = obj_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        else:
            sub_emb, rel_emb, obj_emb, all_ent, corr = self.test_base(self, sub, rel, obj, self.drop, self.drop, mode)
        sim_obj_emb = sub_emb + rel_emb
        #sim_obj_emb和obj_emb的点积
        x = torch.einsum('bkf,bkf->bk', [sim_obj_emb, obj_emb])
        x = torch.sigmoid(x)
        x = x.view(-1, self.p.num_factors)
        if self.p.strategy == 'one_to_n' or neg_ents is None:
            x = torch.einsum('bkf,fkn->bn', [x, all_ent])
            x += self.bias.expand_as(x)
        else:
            x = torch.mul(x.unsqueeze(1), all_ent[neg_ents]).sum(dim=-1)
            x += self.bias[neg_ents]
        pred = torch.sigmoid(x)

        return pred, corr, all_ent


