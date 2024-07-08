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

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


from models import *

from net_util import *

from arg_parser import *


from torch.autograd import Variable
cudnn.benchmark=True


if __name__ == '__main__':


    opts=parse_opts()



    torch.manual_seed(opts.seed)
    if opts.use_gpu:
        if opts.gpu_id < 0:
            opts.gpu_id = 0
        torch.cuda.manual_seed(opts.seed)
        opts.device=torch.device("cuda:%d"%opts.gpu_id)
        print('device: ', opts.device)
    else:
        opts.device=torch.device("cpu")


    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')



    if (opts.dataset == 'cifar10'):
        opts.in_planes = 3
        opts.input_size=32
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        print("| Preparing CIFAR-10 dataset...")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        opts.num_classes = 10
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif (opts.dataset == 'cifar100'):
        opts.in_planes = 3
        opts.input_size = 32
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        print("| Preparing CIFAR-100 dataset...")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        opts.num_classes = 100


    elif opts.dataset=='mnist':
        opts.in_planes=1
        opts.input_size = 28
        trainset= torchvision.datasets.MNIST(root='./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        testset=torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        opts.num_classes = 10
    elif opts.dataset=='stl10':
        opts.in_planes = 3
        opts.input_size = 96
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        print("| Preparing STL10 dataset...")

        trainset = torchvision.datasets.STL10(root='./data',  split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.STL10(root='./data',  split='test', download=True, transform=transform_test)
        opts.num_classes = 10
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opts.batch_size, shuffle=False,
                                              num_workers=opts.num_workers)



    if not os.path.exists(opts.result_path):
        os.mkdir(opts.result_path)

    opts.train_epoch_logger = Logger(os.path.join(opts.result_path, 'train.log'),
                                  ['epoch','time','loss', 'acc', 'extra'])
    opts.train_batch_logger = Logger(os.path.join(opts.result_path, 'train_batch.log'),
                                ['epoch', 'batch', 'loss', 'acc', 'extra'])




    opts.test_epoch_logger = Logger(os.path.join(opts.result_path, 'test.log'),
                        ['epoch','time', 'loss', 'acc','extra'])

    # Model


    print('==> Building model..')

    if opts.arch == 'vgg':
        net = VGG('VGG19')

    if opts.arch=='resnet':
        net = ResNet18()
    if opts.arch=='preact':
        net = PreActResNet18()
    if opts.arch == 'cvnet':
        net = CVNet(input_size=opts.input_size,in_planes=opts.in_planes,out_planes=opts.n_channel,dropout_rate=opts.dropout_rate,num_classes=opts.num_classes, scaling_factor=opts.scaling_factor,growth_rate=opts.growth_rate,append=opts.append)



    # net = GoogLeNet()
    if opts.arch == 'densenet':
        net = densenet_cifar()

    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = CapsNet()

    opts.criterion = nn.CrossEntropyLoss()

    # Training
    if opts.resume_path:
        #Load checkpoint.
        None
        print(opts.resume_path)
        #assert os.path.isdir(opts.resume_path), 'Error: no checkpoint directory found!'
        path=opts.resume_path
        checkpoint = torch.load(opts.resume_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']



    print(opts)

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in parameters])
    print(params,'trainable parameters in the network.')

    set_parameters(opts)

    period = (opts.n_epoch * 2 // 5)

    lr = opts.lr

    for epoch in range(start_epoch, start_epoch + opts.n_epoch):
        opts.epoch = epoch
        if epoch % period == 0:
            if epoch > 0:
                lr /= 10
            print('Current learning rate:', lr)

            parameters = filter(lambda p: p.requires_grad, net.parameters())
            if opts.optimizer == 'SGD':
                opts.current_optimizer = optim.SGD(parameters, lr=lr, momentum=0.9,weight_decay=opts.weight_decay)
            elif opts.optimizer=='Adam':
                opts.current_optimizer = optim.Adam(parameters, lr=lr,weight_decay=opts.weight_decay)
            elif opts.optimizer=='AMSGrad':
                opts.current_optimizer = optim.Adam(parameters, lr=lr, weight_decay=opts.weight_decay,amsgrad=True)


        opts.data_loader = train_loader
        train_net(net,opts)
        opts.data_loader = test_loader

        opts.validating=False
        opts.testing=True

        eval_net(net,opts)

        if (epoch + 1) % opts.checkpoint_epoch == 0:


            plt.subplot(1, 2, 1)
            plt.title('Loss Plot', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.ylabel('Loss', fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            plt.plot(opts.train_losses, 'b')
            if opts.__contains__('test_losses'):
                plt.plot(opts.test_losses, 'r')
            if opts.__contains__('valid_losses'):
                plt.plot(opts.valid_losses, 'g')

            plt.subplot(1, 2, 2)
            plt.title('Accuracy Plot', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.ylabel('Accuracy', fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            plt.plot(opts.train_accuracies, 'b')
            if opts.__contains__('test_accuracies'):
                plt.plot(opts.test_accuracies, 'r')
            if opts.__contains__('valid_accuracies'):
                plt.plot(opts.valid_accuracies, 'g')
            plt.savefig(os.path.join(opts.result_path,'TrainingPlots'))
            plt.clf()
