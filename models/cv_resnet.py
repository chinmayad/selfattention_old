"""
Continuously Variable ResNet
Reference:

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .visualization import *

import math
import numpy as np

class CV_Block(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, scaling_factor=1.,dropout_rate=0.25):
        super(CV_Block, self).__init__()
        self.scaling_factor=scaling_factor
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.dropout1=nn.Dropout2d(dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.dropout2=nn.Dropout2d(dropout_rate)

        if planes>in_planes:
            self.conv_new=nn.Conv2d(in_planes, planes-in_planes, kernel_size=1, padding=0, bias=False)
        else:
            self.conv_new=None

    def forward(self, x):
        # source map size
        N, C, H, W = x.shape

        # target map size
        H_t = math.floor(H * self.scaling_factor)
        W_t = math.floor(W * self.scaling_factor)

        # calculate the coordinates of the sampling grid
        base_grid = x.new_zeros(N, H_t, W_t, 2, requires_grad=False)
        linear_points = torch.from_numpy(np.linspace(-1., 1., W_t)).type_as(base_grid).to(x.device)
        base_grid[..., 0] = linear_points.view(1, 1, -1).repeat(1, H_t, 1).expand_as(base_grid[..., 0])
        linear_points = torch.from_numpy(np.linspace(-1., 1., H_t)).type_as(base_grid).to(x.device)
        base_grid[..., 1] = linear_points.view(1, -1, 1).repeat(1, 1, W_t).expand_as(base_grid[..., 0])

        # sampled map
        x = F.grid_sample(x, base_grid)

        x1 = self.bn1(x)
        if self.conv_new is not None:
            x = torch.cat([x, self.conv_new(x1)], dim=1)
        out = self.conv1(F.relu(x1))#self.dropout1()
        out = self.conv2(F.relu(self.bn2(out)))#self.dropout2()
        out=out + x


        return out


class CV_Bottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, scaling_factor=1.,dropout_rate=0.25):
        super(CV_Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        #self.dropout1=nn.Dropout2d(dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.dropout2=nn.Dropout2d(dropout_rate)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        #self.dropout3=nn.Dropout2d(dropout_rate)


        if self.expansion*planes>in_planes:
            self.conv_new=nn.Conv2d(in_planes, self.expansion*planes-in_planes, kernel_size=1, padding=0, bias=False)
        else:
            self.conv_new=None


    def forward(self, x):
        # source map size
        N, C, H, W = x.shape

        # target map size
        H_t = math.floor(H * self.scaling_factor)
        W_t = math.floor(W * self.scaling_factor)

        # calculate the coordinates of the sampling grid
        base_grid = x.new_zeros(N, H_t, W_t, 2, requires_grad=False)
        linear_points = torch.from_numpy(np.linspace(-1., 1., W_t)).type_as(base_grid).to(x.device)
        base_grid[..., 0] = linear_points.view(1, 1, -1).repeat(1, H_t, 1).expand_as(base_grid[..., 0])
        linear_points = torch.from_numpy(np.linspace(-1., 1., H_t)).type_as(base_grid).to(x.device)
        base_grid[..., 1] = linear_points.view(1, -1, 1).repeat(1, 1, W_t).expand_as(base_grid[..., 0])

        # sampled map
        x = F.grid_sample(x, base_grid)

        x1 = self.bn1(x)
        if self.conv_new is not None:
            x = torch.cat([x, self.conv_new(x1)], dim=1)
        out = self.conv1(F.relu(x1))#self.dropout1()
        out = self.conv2(F.relu(self.bn2(out)))#self.dropout2()
        out = self.conv3(F.relu(self.bn3(out)))#self.dropout3()
        out += x
        return out


class CV_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CV_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], scaling_factor=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], scaling_factor=.5)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], scaling_factor=.5)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], scaling_factor=.5)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, scaling_factor):
        scaling_factors = [scaling_factor] + [1]*(num_blocks-1)
        layers = []
        for scaling_factor in scaling_factors:
            layers.append(block(self.in_planes, planes, scaling_factor))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,msg=None):
        out = self.conv1(x)

        mid=[out]
        out = self.layer1(out)
        mid.append(out)
        out = self.layer2(out)
        mid.append(out)
        out = self.layer3(out)
        mid.append(out)
        out = self.layer4(out)
        mid.append(out)
        out = F.avg_pool2d(out, 4)
        if  msg is not None:
            visualize_maps(mid,msg)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def CV_ResNet18():
    return CV_ResNet(CV_Block, [2,2,2,2])

def CV_ResNet34():
    return CV_ResNet(CV_Block, [3,4,6,3])

def CV_ResNet50():
    return CV_ResNet(CV_Bottleneck, [3,4,6,3])

def CV_ResNet101():
    return CV_ResNet(CV_Bottleneck, [3,4,23,3])

def CV_ResNet152():
    return CV_ResNet(CV_Bottleneck, [3,8,36,3])


def test():
    net = CV_ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
