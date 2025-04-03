import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict


# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
import torch.nn as nn
np.set_printoptions(precision=4)
import torch.fft  # 导入新的fft模块

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


import json
import logging
import logging.config
import os
import sys


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
    config_path = os.path.join(rootpath, config_dir, 'log_config.json')

    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Check if config file is not empty
    if os.path.getsize(config_path) == 0:
        raise ValueError(f"Config file is empty: {config_path}")

    # Try to load the JSON config
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from config file: {config_path}. Error: {e}")

    # Replace invalid characters in the logger name for file safety
    valid_name = name.replace('/', '-').replace(':', '_')

    # Update the log file path in the config
    config_dict['handlers']['file_handler']['filename'] =rootpath+ log_dir + valid_name

    # Configure logging using the config dictionary
    logging.config.dictConfig(config_dict)

    # Create logger
    logger = logging.getLogger(name)

    # Set up console output format and handler
    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


# def get_logger(name, log_dir, config_dir):
#     """
#     Creates a logger object
#
#     Parameters
#     ----------
#     name:           Name of the logger file
#     log_dir:        Directory where logger file needs to be stored
#     config_dir:     Directory from where log_config.json needs to be read
#
#     Returns
#     -------
#     A logger object which writes to both file and stdout
#
#     """
#     config_dict = json.load(open(config_dir + 'log_config.json'))
#     valid_name = name.replace('/', '-').replace(':', '_')  # Replace ':' with '_'
#     config_dict['handlers']['file_handler']['filename'] = log_dir + valid_name
#     logging.config.dictConfig(config_dict)
#     logger = logging.getLogger(name)
#
#     std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
#     consoleHandler = logging.StreamHandler(sys.stdout)
#     consoleHandler.setFormatter(logging.Formatter(std_out_format))
#     logger.addHandler(consoleHandler)
#
#     return logger


def get_combined_results(left_results, right_results):
    results = {}
    count = float(left_results['count'])

    results['left_mr'] = round(left_results['mr'] / count, 5)
    results['left_mrr'] = round(left_results['mrr'] / count, 5)
    #results['left_accuracy'] = round(left_results['accuracy'] / count, 5)
    results['right_mr'] = round(right_results['mr'] / count, 5)
    results['right_mrr'] = round(right_results['mrr'] / count, 5)
    #results['right_accuracy'] = round(right_results['accuracy'] / count, 5)

    #results['accuracy'] = round((left_results['accuracy'] + right_results['accuracy']) / (2 * count), 5)
    results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)

    for k in range(10):
        results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k + 1)] / count, 5)
        results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k + 1)] / count, 5)
        results['hits@{}'.format(k + 1)] = round(
            (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (2 * count), 5)
    return results

def dim_reduction(x, dim):
    '''Reduce the dimension of tensor x to dim by pca'''
    x = x - torch.mean(x, 0, True)
    cov = torch.mm(x.t(), x) / (x.size(0) - 1)
    U, S, V = torch.svd(cov)
    return torch.mm(x, U[:, :dim])

def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param

def com_mult(a, b):
    # 如果a和b是复数形式的
    if a.shape[-1] == 2 and b.shape[-1] == 2:
        r1, i1 = a[..., 0], a[..., 1]
        r2, i2 = b[..., 0], b[..., 1]
        real = r1 * r2 - i1 * i2
        imag = r1 * i2 + i1 * r2
        return torch.stack((real, imag), dim=-1)
    else:
        return a * b

def conj(a):
    # 如果a是复数形式的
    if a.shape[-1] == 2:
        a[..., 1] = -a[..., 1]
    return a

def cconv(a, b):
    return torch.fft.irfft(com_mult(torch.fft.rfft(a, 1), torch.fft.rfft(b, 1)), n=a.shape[-1])

def ccorr(a, b):
    return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a, 1)), torch.fft.rfft(b, 1)), n=a.shape[-1])
