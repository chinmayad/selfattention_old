'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from copy import deepcopy
import numpy as np

import distutils.util



def parse_opts():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--msg', default=False, type=distutils.util.strtobool, help='display message')
    parser.add_argument('--resume_path', default='./checkpoint/best_net.pth', help='resume from checkpoint(''|''./best_models/best_net_0206.pth'')')
    parser.add_argument('--use_gpu', default=torch.cuda.is_available(), type=distutils.util.strtobool, help='Use GPU or not')
    parser.add_argument('--multi_gpu', default=False, type=distutils.util.strtobool, help='Use multi-GPU or not')
    parser.add_argument('--gpu_id', default=-1, type=int, help='Use specific GPU.')
    parser.add_argument('--optimizer', default='Adam', help='optimizer(SGD|Adam)')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--num_workers', default=4, type=int, help='num of fetching threads')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--seed', default=0, type=int, help='randome seed')
    parser.add_argument('--result_path', default='./results', help='result path')
    parser.add_argument('--checkpoint_path', default='./checkpoint', help='checkpoint path')
    parser.add_argument('--matplotlib_mode',default='Agg',help='matplotlib mode (TkAgg|Agg)')#
    parser.add_argument('--dataset', default='cifar10', help='dataset(cifar10|cifar100)')
    parser.add_argument('--checkpoint_epoch', default=20, type=int, help='epochs to save checkpoint ')
    parser.add_argument('--n_epoch', default=350, type=int, help='training epochs')
    parser.add_argument('--arch', default='cvnet', help='architecture')
    parser.add_argument('--visualize','--save_plot', default=False, type=distutils.util.strtobool,  help='visualize feature maps')
    parser.add_argument('--viz_T', default=100, type=int,  help='visualization period')

    parser.add_argument('--append', default=False, type=distutils.util.strtobool, help='append output channels as in densenet')
    parser.add_argument('--n_channel','--n_slice', default=128, type=int, help='feature channels.')
    parser.add_argument('--growth_rate', default=64, type=int, help='feature channel growth rate.')
    parser.add_argument('--scaling_factor','--scaling_rate', default=3./4., type=float, help='scaling factor of each continuously variable layer')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate')

    args = parser.parse_args()

    return args
