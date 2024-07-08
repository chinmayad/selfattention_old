import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np



import matplotlib

matplotlib.use('Agg')

from .visualization import *
from models.deformable_models import  DeformConv2D,DeformRouting

def scaling(maps,scaling_factor):
    N, C, H, W = maps.shape

    # target map size
    H_t = round(H * scaling_factor)
    W_t = round(W * scaling_factor)

    pool_size=math.floor(1./scaling_factor)
    if pool_size>=2:
        maps=F.avg_pool2d(maps,pool_size,ceil_mode=True)
        N, C, H, W = maps.shape

    if H!=H_t or W!=W_t:
        # calculate the coordinates of the sampling grid
        base_grid = maps.new_zeros(N, H_t, W_t, 2, requires_grad=False)
        linear_points = torch.from_numpy(np.linspace(-1., 1., W_t)).type_as(base_grid).to(maps.device)
        base_grid[..., 0] = linear_points.view(1, 1, -1).repeat(1, H_t, 1).expand_as(base_grid[..., 0])
        linear_points = torch.from_numpy(np.linspace(-1., 1., H_t)).type_as(base_grid).to(maps.device)
        base_grid[..., 1] = linear_points.view(1, -1, 1).repeat(1, 1, W_t).expand_as(base_grid[..., 0])
        # sampled map
        maps = F.grid_sample(maps, base_grid)

    return maps



class ContinuouslyVariableLayer(nn.Module):
    """
    This function continuously samples the feature maps
    """
    def __init__(self, in_planes, out_planes,append,kernel_size=3,padding=1,dropout_rate=0.25,scaling_factor=0.75):
        super(ContinuouslyVariableLayer, self).__init__()
        self.scaling_factor=scaling_factor
        self.dropout_rate=dropout_rate
        self.append=append
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        if dropout_rate>0.:
            self.dropout1 = nn.Dropout2d(dropout_rate)
            self.dropout2 = nn.Dropout2d(dropout_rate)

    def forward(self, x):

        if self.dropout_rate>0.:
            out = self.conv1(self.dropout1(F.relu(self.bn1(x))))
            out = self.conv2(self.dropout2(F.relu(self.bn2(out))))
        else:
            out = self.conv1(F.relu(self.bn1(x)))
            out = self.conv2(F.relu(self.bn2(out)))

        if not self.append:
            assert out.shape[1] >= x.shape[1]
            if out.shape[1] > x.shape[1]:
                C=x.shape[1]
                out=torch.cat([out[:,:C]+x,out[:,C:]],dim=1)
            else:
                out=out+x
        else:
            out = torch.cat([x, out], dim=1)


        if self.scaling_factor!=1.:
            out=scaling(out,self.scaling_factor)


        return out





class CVNet(nn.Module):
    def __init__(self,input_size , in_planes=3, out_planes=32,scaling_factor=0.75,growth_rate=64, dropout_rate=0.25, final_map_size=4, num_classes=10,append=False):
        super(CVNet, self).__init__()
        self.scaling_factor = scaling_factor
        self.append=append
        self.growth_rate=growth_rate
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.final_map_size=final_map_size

        self.layers = nn.ModuleList()
        output_size=input_size
        while output_size > final_map_size:
            if append:
                #like densenet
                new_out_planes = growth_rate
            else:
                #densenet+resnet
                new_out_planes =out_planes + growth_rate
            self.layers.append(ContinuouslyVariableLayer(in_planes=out_planes, out_planes=new_out_planes,append=append,dropout_rate=dropout_rate,scaling_factor=scaling_factor))
            output_size = math.floor(output_size * scaling_factor)
            out_planes = out_planes + growth_rate

        print(len(self.layers), ' middle layers.')
        print(out_planes, ' final feature maps.')
        self.conv2 = nn.Conv2d(out_planes, 16, kernel_size=1, stride=1, padding=0, bias=False)  # to reduce memory size
        self.bn2 = nn.BatchNorm2d(16)
        self.offsets = nn.Conv2d(16, 18, kernel_size=3, padding=1, stride=1)  # 18= kernal_size^2*2
        self.conv3 = DeformRouting(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_planes)

        self.bn=nn.BatchNorm2d(out_planes)
        self.linear = nn.Linear(out_planes, num_classes)

    def forward(self, x,targets=None,msg=None):
        out = self.conv1(x)

        mid=[out]
        feats_vol=out
        for i, layer in enumerate(self.layers):
            mid.append(layer(mid[-1]))

        out=mid[-1]
        out = F.relu(self.bn2(self.conv2(out)))
        offsets = self.offsets(out)
        out = F.relu(self.bn3(self.conv3(out, offsets)))
        out = self.conv4(out)

        if (msg is not None) and (not self.append):
            visualize_all_maps(mid,msg)

        pool_size=out.shape[2:]
        out = F.avg_pool2d(F.relu(self.bn(out)), pool_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out






