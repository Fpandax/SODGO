import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict

import pandas as pd
from ordered_set import OrderedSet

# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
import torch.nn as nn
np.set_printoptions(precision=4)


def set_gpu(gpus):
    """
    Sets the GPU to be used for the run

    Parameters
    ----------
    gpus:           List of GPUs to be used for the run

    Returns
    -------

    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def get_logger(name, log_dir, config_dir, rootpath):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

    """
    config_dict = json.load(open(rootpath+ config_dir + 'log_config.json'))
    valid_name = name.replace('/', '-').replace(':', '_')  # Replace ':' with '_'
    config_dict['handlers']['file_handler']['filename'] =rootpath + log_dir + valid_name
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

def load_id(rootpath, dataset, type):
    """
    Loads the entity/relation ids

    Parameters
    ----------
    datapath:       Path to the dataset

    Returns
    -------
    entity2id:      Mapping of entity to id
    relation2id:    Mapping of relation to id

    """
    if type=='entity':
        ent2id = {}
        with open('{}/data/{}/prior/entity2id.txt'.format(rootpath, dataset)) as f:
            for line in f:
                ent, idx = line.strip().split('\t')
                ent2id[ent] = int(idx)
        return ent2id
    if type=='relation':
        rel2id = {}
        with open('{}/data/{}/prior/relation2id.txt'.format(rootpath, dataset)) as f:
            for line in f:
                rel, idx = line.strip().split('\t')
                rel2id[rel] = int(idx)
        return rel2id
    else:
        raise ValueError('Invalid type')

def load_node_type(file_path):
    node_set = OrderedSet()
    node_type = {}
    with open(file_path) as f:
        for line in f:
            id, type = line.strip().split('\t')
            node_set.add(type)
    type2id = {type: idx for idx, type in enumerate(node_set)}

    with open(file_path) as f:
        for line in f:
            id, type = line.strip().split('\t')
            node_type[id] = type2id[type]

    return node_type, type2id


    return go_type

def load_go_namespace(file_path):
    go_namespace = {}
    namespace_set = set()   
    with open(file_path) as f:
        for line in f:
            go_id, namespace = line.strip().split('\t')
            namespace_set.add(namespace)   
    namespace2id = {namespace: idx for idx, namespace in enumerate(namespace_set)}

    with open(file_path) as f:
        for line in f:
            go_id, namespace = line.strip().split('\t')
            go_namespace[go_id] = namespace2id[namespace]

    return go_namespace, namespace2id

def load_goemb(file_path, entity2id):
    go_df = pd.read_csv(file_path)
     
    go_emb = {}
    for _, row in go_df.iterrows():
        go_id = row['GO_id']
        if go_id in entity2id:
            go_emb[entity2id[go_id]] = row['reduced_embedding']
    return go_emb


# def get_goenembed(goemb, entityemb):
 
 
#
 
#     new_entityemb = entityemb.clone()
#     for go_id, embedding in goemb.items():
 
#         if isinstance(embedding, str):
#             embedding = [float(x) for x in embedding.strip("[]").split(",")]
#
 
#         embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
#
 
#         if embedding_tensor.size(0) != embedding_dim:
#             raise ValueError(f"GO embedding for ID {go_id} has incorrect dimension: {embedding_tensor.size(0)}")
#
 
#         new_entityemb[go_id] = embedding_tensor
#
 
#     return Parameter(new_entityemb)

def initialize_embedding(goemb, shape):
     
    entityemb = Parameter(torch.Tensor(*shape))
    xavier_normal_(entityemb.data)

     
    embedding_dim = entityemb.size(1)

     
    for go_id, embedding in goemb.items():

         
        if isinstance(embedding, str):
            embedding = [float(x) for x in embedding.strip("[]").split(",")]
         
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        if embedding_tensor.size(0) != embedding_dim:
            raise ValueError(f"GO embedding for ID {go_id} has incorrect dimension: {embedding_tensor.size(0)}")

         
        entityemb.data[go_id] = embedding_tensor

    return entityemb




def init_subweight(num_sub, num_factors):
    sub_weight = torch.zeros(num_sub, num_factors)
     
    for i in range(num_sub-1):
        sub_weight[i][i] = 1
    return sub_weight


def get_combined_results(left_results, right_results):
    results = {}
    count = float(left_results['count'])

    results['left_mr'] = round(left_results['mr'] / count, 5)
    results['left_mrr'] = round(left_results['mrr'] / count, 5)
    results['right_mr'] = round(right_results['mr'] / count, 5)
    results['right_mrr'] = round(right_results['mrr'] / count, 5)
    results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)

    for k in range(10):
        results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k + 1)] / count, 5)
        results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k + 1)] / count, 5)
        results['hits@{}'.format(k + 1)] = round(
            (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (2 * count), 5)
    return results


def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def cconv(a, b):
    return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
