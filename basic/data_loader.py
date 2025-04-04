from helper import *
from torch.utils.data import Dataset
import pandas as pd
import torch

def load_node_attributes(file_path, ent2id):
    go_df = pd.read_csv(file_path)
    node_attributes = {}
    node_ebedding = {}
    namespace2id = {namespace: idx for idx, namespace in enumerate(go_df['namespace'].unique())}


    for _, row in go_df.iterrows():
        go_id = row['GO_id']  
        if go_id in ent2id:
            node_attributes[ent2id[go_id]] = namespace2id[row['namespace']]
            node_ebedding[ent2id[go_id]] = row['embedding']

    return node_attributes, namespace2id, node_ebedding

def load_relationship_strengths(file_path, namespace2id, rel2id):
    rel_df = pd.read_csv(file_path)
    rel_strength = {}

    for _, row in rel_df.iterrows():
        namespace1_id = namespace2id.get(row['namespace1'])
        namespace2_id = namespace2id.get(row['namespace2'])
        rel_id = rel2id.get(row['relationship'])

        if namespace1_id is not None and namespace2_id is not None and rel_id is not None:
            rel_strength[(namespace1_id, namespace2_id, rel_id)] = row['count']

            reverse_rel_id = rel2id.get(row['relationship'] + '_reverse')
            if reverse_rel_id is not None:
                rel_strength[(namespace2_id, namespace1_id, reverse_rel_id)] = row['count']

    
    for ns1 in namespace2id.values():
        for ns2 in namespace2id.values():
            for rel in rel2id.values():
                if (ns1, ns2, rel) not in rel_strength:
                    rel_strength[(ns1, ns2, rel)] = 0

    return rel_strength





class TrainDataset(Dataset):
    """
    Training Dataset class.

    Parameters
    ----------
    triples:    The triples used for training the model
    params:     Parameters for the experiments

    Returns
    -------
    A training Dataset class instance used by DataLoader
    """

    def __init__(self, triples, params, node_attributes, rel_strength):
        self.triples = triples
        self.p = params
        self.entities = np.arange(self.p.num_ent, dtype=np.int32)
        self.node_attributes = node_attributes
        self.rel_strength = rel_strength


    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label, sub_samp = torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
        trp_label = self.get_label(label)

        if self.p.lbl_smooth != 0.0:
            trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0 / self.p.num_ent)

        if self.p.strategy == 'one_to_n':
            orig_label = self.get_label(label)
            return triple, trp_label, orig_label, None, None

        elif self.p.strategy == 'one_to_x':
            sub_samp = torch.FloatTensor([sub_samp])
            neg_ent = torch.LongTensor(self.get_neg_ent(triple, label))
            return triple, trp_label, neg_ent, sub_samp
        else:
            raise NotImplementedError

    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        orig_label = torch.stack([_[2] for _ in data], dim=0)
        # triple: (batch-size) * 3(sub, rel, -1) trp_label (batch-size) * num entity
        # return triple, trp_label
        if not data[0][3] is None:  # one_to_x
            neg_ent = torch.stack([_[2] for _ in data], dim=0)
            sub_samp = torch.cat([_[3] for _ in data], dim=0)
            return triple, trp_label, neg_ent, sub_samp
        else:
            return triple, trp_label, orig_label

    # def get_neg_ent(self, triple, label):
    #     def get(triple, label):
    #         pos_obj = label
    #         mask = np.ones([self.p.num_ent], dtype=bool)
    #         mask[label] = 0
    #         neg_ent = np.int32(
    #             np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
    #         neg_ent = np.concatenate((pos_obj.reshape([-1]), neg_ent))
    #
    #         return neg_ent
    #
    #     neg_ent = get(triple, label)
    #     return neg_ent
    def get_neg_ent(self, triple, label):
        def get(triple, label):
            if self.p.strategy == 'one_to_x':
                pos_obj = triple[2]
                mask = np.ones([self.p.num_ent], dtype=bool)
                mask[label] = 0
                neg_ent = np.int32(np.random.choice(self.entities[mask], self.p.neg_num, replace=False)).reshape([-1])
                neg_ent = np.concatenate((pos_obj.reshape([-1]), neg_ent))
            else:
                pos_obj = label
                mask = np.ones([self.p.num_ent], dtype=bool)
                mask[label] = 0
                neg_ent = np.int32(
                    np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
                neg_ent = np.concatenate((pos_obj.reshape([-1]), neg_ent))

                if len(neg_ent) > self.p.neg_num:
                    import pdb;
                    pdb.set_trace()

            return neg_ent

        neg_ent = get(triple, label)
        return neg_ent

    def get_label(self, label):
        # y = np.zeros([self.p.num_ent], dtype=np.float32)
        # for e2 in label: y[e2] = 1.0
        # return torch.FloatTensor(y)
        if self.p.strategy == 'one_to_n':
            y = np.zeros([self.p.num_ent], dtype=np.float32)
            for e2 in label: y[e2] = 1.0
        elif self.p.strategy == 'one_to_x':
            y = [1] + [0] * self.p.neg_num
        else:
            raise NotImplementedError
        return torch.FloatTensor(y)


class TestDataset(Dataset):
    """
    Evaluation Dataset class.

    Parameters
    ----------
    triples:    The triples used for evaluating the model
    params:     Parameters for the experiments
    node_attributes: 节点属性
    rel_strength: 关系强度

    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation
    """

    def __init__(self, triples, params, node_attributes, rel_strength):
        self.triples = triples
        self.p = params
        self.node_attributes = node_attributes
        self.rel_strength = rel_strength

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])
        label = self.get_label(label)

        return triple, label

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)
