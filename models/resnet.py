"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from models.deformable_models import  DeformConv2D,DeformRouting

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_deformable(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_deformable, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.offsets = nn.Conv2d(planes, 18, kernel_size=3, padding=1,stride=stride) #18= kernal_size^2*2
        self.conv2 = DeformConv2D(planes, planes, kernel_size=3, padding=1,bias=False)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        offsets = self.offsets(out)
        out = self.bn2(self.conv2(out,offsets))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_deformable_attention(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_deformable_attention, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, 16, kernel_size=1, stride=1, padding=0, bias=False)#to reduce memory size
        self.bn2 = nn.BatchNorm2d(16)
        self.offsets = nn.Conv2d(16, 18, kernel_size=3, padding=1,stride=stride) #18= kernal_size^2*2
        self.conv3 = DeformRouting(16, 16, kernel_size=3, padding=1,bias=False,stride=stride)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = F.relu(self.bn2(self.conv2(out)))
        offsets = self.offsets(out)
        out = F.relu(self.bn3(self.conv3(out, offsets)))
        out = self.bn4(self.conv4(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck_deformable(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.offsets = nn.Conv2d(planes, 18, kernel_size=3, padding=1,stride=stride)  # 18= kernal_size^2*2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        offsets = self.offsets(out)
        out = F.relu(self.bn2(self.conv2(out, offsets)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_deformable_attention(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_deformable_attention, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.offsets = nn.Conv2d(planes, 18, kernel_size=3, padding=1,stride=stride)  # 18= kernal_size^2*2
        self.conv2 = nn.Conv2d(planes, 16, kernel_size=1, stride=1, padding=0, bias=False)#to reduce memory size
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = DeformRouting(16, 16, kernel_size=3, padding=1,bias=False,stride=stride)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        offsets = self.offsets(out)
        out = F.relu(self.bn2(self.conv2(out)))

        out = F.relu(self.bn3(self.conv3(out,offsets)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.bn5(self.conv5(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


import numpy as np

def downsample_and_cat(x,mid):
    N,C,H_t,W_t=x.shape
    # calculate the coordinates of the sampling grid
    base_grid = x.new_zeros(N, H_t, W_t, 2, requires_grad=False)

    linear_points = torch.from_numpy(np.linspace(-1., 1., W_t)).type_as(base_grid).to(x.device)
    base_grid[..., 0] = linear_points.view(1, 1, -1).repeat(1, H_t, 1).expand_as(base_grid[..., 0])
    linear_points = torch.from_numpy(np.linspace(-1., 1., H_t)).type_as(base_grid).to(x.device)
    base_grid[..., 1] = linear_points.view(1, -1, 1).repeat(1, 1, W_t).expand_as(base_grid[..., 0])

    x0=[]
    for i,m in enumerate(mid):
        x0.append(F.grid_sample(m, base_grid))
    x0=torch.cat(x0,dim=1)
    return x0



class Bottleneck_deformable_global_attention(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, global_in_planes,stride=1):
        super(Bottleneck_deformable_global_attention, self).__init__()
        self.conv1 = nn.Conv2d(in_planes+global_in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.offsets = nn.Conv2d(planes, 18, kernel_size=3, padding=1, stride=stride)  # 18= kernal_size^2*2
        self.conv2 = nn.Conv2d(planes, 16, kernel_size=1, stride=1, padding=0, bias=False)  # to reduce memory size
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = DeformRouting(16, 16, kernel_size=3, padding=1, bias=False, stride=stride)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.expansion * planes)



        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x,mid):
        x0=downsample_and_cat(x, mid)
        out = F.relu(self.bn1(self.conv1(torch.cat((x,x0),dim=1))))
        offsets = self.offsets(out)
        out = F.relu(self.bn2(self.conv2(out)))

        out = F.relu(self.bn3(self.conv3(out, offsets)))

        out = F.relu(self.bn4(self.conv4(out)))
        out = self.bn5(self.conv5(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#        self.layer4 = self._make_layer(BasicBlock_deformable_attention, 512, num_blocks[3], stride=2)
        #self.layer4 = self._make_layer(BasicBlock_deformable, 512, num_blocks[3], stride=2)
       #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer4 = self._make_layer2(Bottleneck_deformable_global_attention, planes=512,global_in_planes=64+64+128+256, num_blocks=num_blocks[3], stride=2)
        self.linear = nn.Linear(512*Bottleneck_deformable_global_attention.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes,global_in_planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,global_in_planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, x):
        mid=[]
        out = F.relu(self.bn1(self.conv1(x)))
        mid.append(out)
        out = self.layer1(out)
        mid.append(out)
        out = self.layer2(out)
        mid.append(out)
        out = self.layer3(out)
        mid.append(out)
        for i,layer in enumerate(self.layer4):
            out = layer(out,mid)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
