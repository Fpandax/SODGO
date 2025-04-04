from helper import *

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch.nn as nn
import torch


class DisenLayer(MessagePassing):
    def __init__(self, edge_index, edge_type, in_channels, out_channels, num_rels, num_namespaces,
                 node_type=None, go_namespace=None, type2id=None,
                 act=lambda x: x, params=None, head_num=8, gfsa_order=2):
        """
        初始化函数
        :param edge_index: 边的索引
        :param edge_type: 边的类型
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param num_rels: 关系数量
        :param act: 激活函数，默认为恒等函数
        :param params: 参数
        :param head_num: 头部数量
        """
        super(self.__class__, self).__init__(aggr='add', flow='target_to_source', node_dim=0)

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.device = None
        self.head_num = head_num
        self.num_rels = num_rels
        self.num_factors = self.p.num_factors
        self.type2id = type2id
        self.num_namespaces = num_namespaces

         
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm1d(self.p.num_factors * out_channels)   
        if self.p.bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))

        num_edges = self.edge_index.size(1) // 2
        if self.device is None:
            self.device = self.edge_index.device
         
        self.in_index, self.out_index = self.edge_index[:, :num_edges], self.edge_index[:, num_edges:]
        self.in_type, self.out_type = self.edge_type[:num_edges], self.edge_type[num_edges:]
        self.loop_index = torch.stack([torch.arange(self.p.num_ent), torch.arange(self.p.num_ent)]).to(self.device)
        self.loop_type = torch.full((self.p.num_ent,), 2 * self.num_rels, dtype=torch.long).to(self.device)

         
        # self.node_type = torch.zeros(self.p.num_ent, dtype=torch.long).to(self.device)
        # for node_id in range(self.p.num_ent):
        #     if node_id in node_type:
        #         self.node_type[node_id] = torch.tensor(node_type[node_id], dtype=torch.long).to(self.device)
        #     else:
        #         self.node_type

         
        self.go_namespace = torch.zeros(self.p.num_ent, dtype=torch.long).to(self.device)
        for go_id in range(self.p.num_ent):
            if go_id in go_namespace:
                self.go_namespace[go_id] = torch.tensor(go_namespace[go_id], dtype=torch.long).to(self.device)
            else:
                # print(f"Warning: go_id {go_id} is missing in go_namespace. Initializing to 3.")
                self.go_namespace[go_id] = 3

        self.leakyrelu = nn.LeakyReLU(0.2)
        if self.p.att_mode == 'cat_emb' or self.p.att_mode == 'cat_weight':
            self.att_weight = get_param((1, self.p.num_factors, 2 * out_channels))
        else:   
            self.att_weight = get_param((1, self.p.num_factors, out_channels))  # [1,3,200]
        self.rel_weight = get_param((2 * self.num_rels + 1, self.p.num_factors, out_channels))  # [21,3,200]
        self.loop_rel = get_param((1, out_channels))  # [1,200]
        self.w_rel = get_param((out_channels, out_channels))  # [200,200]
         
        self.space_weight = init_subweight(self.num_namespaces+1, self.num_factors)  

        self.gfsa_order = gfsa_order
        self.w_0, self.w_1, self.w_K = initialize_gfsa_params(self.p.num_factors)


    def forward(self, x, rel_embed, mode):
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        edge_index = torch.cat([self.edge_index, self.loop_index], dim=1)
        edge_type = torch.cat([self.edge_type, self.loop_type])

         
        x = self.propagate(edge_index, size=None, x=x, edge_type=edge_type, rel_embed=rel_embed,
                           rel_weight=self.rel_weight,
                            go_namespace=self.go_namespace, type2id=self.type2id)
        if self.p.bias:
            x = x + self.bias

        x = self.bn(x.view(-1, self.p.num_factors * self.p.gcn_dim)).view(-1, self.p.num_factors, self.p.gcn_dim)
        entity = x if self.p.no_act else self.act(x)

        return entity, torch.matmul(rel_embed, self.w_rel)[:-1]

    # def message(self, x_j: Tensor) -> Tensor:
    #     print("'message' is called")
    #     return x_j

    def message(self, edge_index_i, edge_index_j, x_i, x_j, edge_type, rel_embed, rel_weight, go_namespace, type2id):

         
        go_namespace_j = torch.index_select(go_namespace.to(self.device), 0, edge_index_j.to(self.device))

         
        rel_embed = torch.index_select(rel_embed.to(self.device), 0, edge_type.to(self.device))  # 158254,200
        rel_weight = torch.index_select(rel_weight.to(self.device), 0, edge_type.to(self.device))  # (158254,3,200)

         
        xj_rel = self.rel_transform(x_j.to(self.device), rel_embed, rel_weight)

         
        selected_space_weight = torch.index_select(self.space_weight.to(self.device), 0, go_namespace_j.to(self.device))

         
        alpha = self._get_attention(edge_index_i, edge_index_j, x_i, x_j, rel_embed, rel_weight, xj_rel)
        alpha = torch.einsum('ek,ekf->ekf', [selected_space_weight, alpha])

         
        alpha = self.drop(alpha)
        assert xj_rel is not None, "xj_rel is None"
        assert alpha is not None, "alpha is None"

        return xj_rel * alpha

        return xj_rel * gfsa_out

    def update(self, aggr_out):
        # print("'update' is called")
        return aggr_out

    def _get_attention(self, edge_index_i, edge_index_j, x_i, x_j, rel_embed, rel_weight, mes_xj):
        if self.p.att_mode == 'learn':
            alpha = self.leakyrelu(torch.einsum('ekf, xkf->ek', [mes_xj, self.att_weight]))  # [E K]
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)
        elif self.p.att_mode == 'dot_weight':
            sub_rel_emb = x_i * rel_weight
            obj_rel_emb = x_j * rel_weight

            alpha = self.leakyrelu(torch.einsum('ekf,ekf->ek', [sub_rel_emb, obj_rel_emb]))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)

        elif self.p.att_mode == 'dot_emb':
            sub_rel_emb = x_i * rel_embed.unsqueeze(1)
            obj_rel_emb = x_j * rel_embed.unsqueeze(1)

            alpha = self.leakyrelu(torch.einsum('ekf,ekf->ek', [sub_rel_emb, obj_rel_emb]))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)

        elif self.p.att_mode == 'cat_weight':
            sub_rel_emb = x_i * rel_weight
            obj_rel_emb = x_j * rel_weight

            alpha = self.leakyrelu \
                (torch.einsum('ekf,xkf->ek', torch.cat([sub_rel_emb, obj_rel_emb], dim=2), self.att_weight))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)

        elif self.p.att_mode == 'cat_emb':
            sub_rel_emb = x_i * rel_embed.unsqueeze(1)
            obj_rel_emb = x_j * rel_embed.unsqueeze(1)

            alpha = self.leakyrelu \
                (torch.einsum('ekf,xkf->ek', torch.cat([sub_rel_emb, obj_rel_emb], dim=2), self.att_weight))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)
        else:
            raise NotImplementedError
        return alpha.unsqueeze(2)

    def rel_transform(self, ent_embed, rel_embed, rel_weight, opn=None):
        if opn is None:
            opn = self.p.opn
        if opn == 'corr':
            trans_embed = ccorr(ent_embed * rel_weight, rel_embed.unsqueeze(1))
        elif opn == 'corr_ra':
            trans_embed = ccorr(ent_embed * rel_weight, rel_embed)
        elif opn == 'sub':
            trans_embed = ent_embed * rel_weight - rel_embed.unsqueeze(1)
        elif opn == 'es':
            trans_embed = ent_embed
        elif opn == 'sub_ra':
            trans_embed = ent_embed * rel_weight - rel_embed.unsqueeze(1)
        elif opn == 'mult':
            trans_embed = (ent_embed * rel_embed.unsqueeze(1)) * rel_weight
        elif opn == 'mult_ra':
            trans_embed = (ent_embed * rel_embed) * rel_weight
        elif opn == 'cross':
            trans_embed = ent_embed * rel_embed.unsqueeze(1) * rel_weight + ent_embed * rel_weight
        elif opn == 'cross_wo_rel':
            trans_embed = ent_embed * rel_weight
        elif opn == 'cross_simplfy':
            trans_embed = ent_embed * rel_embed + ent_embed
        elif opn == 'concat':
            trans_embed = torch.cat([ent_embed, rel_embed], dim=1)
        elif opn == 'concat_ra':
            trans_embed = torch.cat([ent_embed, rel_embed], dim=1) * rel_weight
        elif opn == 'ent_ra':
            trans_embed = ent_embed * rel_weight + rel_embed
        else:
            raise NotImplementedError
        return trans_embed  # (158254,3,200)

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
