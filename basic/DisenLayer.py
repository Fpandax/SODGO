from helper import *
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch.nn as nn
import torch.fft

analy_alpha = None

class DisenLayer(MessagePassing):
    def __init__(self, num_ent, edge_index, edge_type, in_channels, out_channels, num_rels, act=lambda x: x,
                 params=None, head_num=8, node_attributes=None, rel_strength=None):
        super(self.__class__, self).__init__(aggr='add', flow='source_to_target', node_dim=0)

        self.num_ent = num_ent
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

        self.node_attributes = torch.zeros((num_ent), dtype=torch.long)
        for node_id in range(num_ent):
            if node_id in node_attributes:
                self.node_attributes[node_id] = torch.tensor(node_attributes[node_id], dtype=torch.long)
            else:
                print(f"Warning: node_id {node_id} is missing in node_attributes. Initializing to zero.")

        self.stats = rel_strength
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm1d(self.num_factors * out_channels)
        if self.p.bias:
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

        num_edges = self.edge_index.size(1) // 2
        if self.device is None:
            self.device = self.edge_index.device
        self.in_index, self.out_index = self.edge_index[:, :num_edges], self.edge_index[:, num_edges:]
        self.in_type, self.out_type = self.edge_type[:num_edges], self.edge_type[num_edges:]
        self.loop_index = torch.stack([torch.arange(self.p.num_ent), torch.arange(self.p.num_ent)]).to(self.device)
        self.loop_type = torch.full((self.p.num_ent,), 2 * self.num_rels, dtype=torch.long).to(self.device)

        self.node_attributes = self.node_attributes.to(self.device)


        self.leakyrelu = nn.LeakyReLU(0.2)

        self.att_weight = get_param((1, self.num_factors, 2 * out_channels))
        self.rel_weight = get_param((2 * self.num_rels + 1, self.num_factors, out_channels))
        self.loop_rel = get_param((1, out_channels))
        self.w_rel = get_param((out_channels, out_channels))
        self.space_weight = nn.Parameter(torch.eye(3, self.num_factors))


    def forward(self, x, rel_embed, mode):
        # 将 rel_embed 和 edge_index 等移到与模型同一设备上
        rel_embed = torch.cat([rel_embed, self.loop_rel.to(self.device)], dim=0)
        edge_index = torch.cat([self.edge_index.to(self.device), self.loop_index.to(self.device)], dim=1)
        edge_type = torch.cat([self.edge_type.to(self.device), self.loop_type.to(self.device)])

        # 进行传播计算时，确保所有张量位于相同设备
        out= self.propagate(edge_index, size=None,
                             x=x.to(self.device), edge_type=edge_type.to(self.device),
                             rel_embed=rel_embed.to(self.device), rel_weight=self.rel_weight.to(self.device),
                             node_attributes=self.node_attributes.to(self.device))

        if self.p.bias:
            out = out + self.bias.to(self.device)

        out = self.bn(out.view(-1, self.num_factors * self.p.gcn_dim)).view(-1, self.num_factors, self.p.gcn_dim)
        entity1 = out if self.p.no_act else self.act(out)
        global analy_alpha
        return entity1, torch.matmul(rel_embed, self.w_rel.to(self.device))[:-1], analy_alpha

    def message(self, edge_index_i, edge_index_j, x_i, x_j, edge_type, rel_embed, rel_weight, node_attributes):
        node_attr_i = self.node_attributes[edge_index_i.to(self.device)]
        node_attr_j = self.node_attributes[edge_index_j.to(self.device)]

        rel_embed = torch.index_select(rel_embed.to(self.device), 0, edge_type.to(self.device))
        rel_weight = torch.index_select(rel_weight.to(self.device), 0, edge_type.to(self.device))

        xj_rel = self.rel_transform(x_j, rel_embed, rel_weight)

        valid_subspace = (node_attr_j.unsqueeze(1) == torch.arange(self.num_factors).to(node_attr_j.device)).float()
        selected_space_weight = torch.matmul(valid_subspace, self.space_weight.T)

        alpha = self._get_attention(edge_index_i, edge_index_j, x_i, x_j, rel_embed, rel_weight, xj_rel)
        global analy_alpha
        analy_alpha = alpha * selected_space_weight.unsqueeze(2)
        alpha = alpha * selected_space_weight.unsqueeze(2)

        alpha = self.drop(alpha)

        return xj_rel * alpha

    def _get_attention(self, edge_index_i, edge_index_j, x_i, x_j, rel_embed, rel_weight, mes_xj):

            sub_rel_emb = x_i * rel_embed.unsqueeze(1)
            obj_rel_emb = x_j * rel_embed.unsqueeze(1)

            alpha = self.leakyrelu(
                    torch.einsum('ekf,xkf->ek', torch.cat([sub_rel_emb, obj_rel_emb], dim=2), self.att_weight))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)
            return alpha.unsqueeze(2)

    def rel_transform(self, ent_embed, rel_embed, rel_weight, opn=None):
        trans_embed = ent_embed * rel_embed.unsqueeze(1) * rel_weight + ent_embed * rel_weight
        return trans_embed

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
