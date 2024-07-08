import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np


class DeformConv2D(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size=3, padding=1, bias=None, grid_kernel_size=1, grid_conv_pad=0):
        super(DeformConv2D, self).__init__()
        self.grid_gen = nn.Conv2d(channels_in, kernel_size ** 2 * 2, kernel_size=grid_kernel_size, stride=1,
                                   padding=grid_conv_pad)
        self.grid_gen.weight.data.fill_(0)
        self.grid_gen.bias.data.fill_(0)
        #self.grid_gen.weight.requires_grad = False
        #self.grid_gen.bias.requires_grad = False
        self.deform_conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=kernel_size, padding=0,
                                     bias=bias)
        self.kernel_size = kernel_size

    def forward(self, inputs, offset=None):
        N, C, H, W = inputs.shape
        kH = self.kernel_size
        kW = self.kernel_size

        if offset is None:
            offset = self.grid_gen(inputs)

        # calculate the coordinates of the sampling grid
        base_grid = inputs.new_zeros(1, H, W, 1, 1, 2,requires_grad=False)
        linear_points = torch.from_numpy(np.linspace(-1., 1., W)).type_as(base_grid).to(inputs.device)
        base_grid[..., 0] = torch.ger(inputs.new_ones(H,requires_grad=False), linear_points).view(1, H, W, 1, 1)
        linear_points = torch.from_numpy(np.linspace(-1., 1., H)).type_as(base_grid).to(inputs.device)
        base_grid[..., 1] = torch.ger(linear_points, inputs.new_ones(W,requires_grad=False)).view(1, H, W, 1, 1)

        # calculate the offset for each kernel entry
        kernel_offset = inputs.new_zeros(1, 1, 1, kH, kW, 2,requires_grad=False)
        linear_points = torch.from_numpy(np.linspace(-(kW - 1) / (W - 1), (kW - 1) / (W - 1), kW)).type_as(
            kernel_offset).to(inputs.device)
        kernel_offset[..., 0] = torch.ger(inputs.new_ones(kH,requires_grad=False), linear_points).view(1, 1, 1, kH, kW)
        linear_points = torch.from_numpy(np.linspace(-(kH - 1) / (H - 1), (kH - 1) / (H - 1), kH)).type_as(
            kernel_offset).to(inputs.device)
        kernel_offset[..., 1] = torch.ger(linear_points, inputs.new_ones(kW,requires_grad=False)).view(1, 1, 1, kH, kW)

        # (1,H,W, kH, kW,2)
        base_grid = Variable(base_grid + kernel_offset, requires_grad=False)

        # (N,H,W, kH, kW,2)
        offset = offset.permute(0, 2, 3, 1).contiguous().view(N, H, W, kH, kW, 2)

        grid = offset + base_grid

        # (N,2,W,kW,H*kH)
        grid = grid.permute(0, 5, 2, 4, 1, 3).contiguous().view(N, 2, W, kW, H * kH)
        # (N, 2, H*kH, W, kW)  -> (N, H*kH, W*kW, 2)
        grid = grid.permute(0, 1, 4, 2, 3).contiguous().view(N, 2, H * kH, W * kW).permute(0, 2, 3, 1)

        input_wrapped = F.grid_sample(inputs, grid)  # ,padding_mode='border'

        output = self.deform_conv(input_wrapped)

        return output


class DeformRouting(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size=3, padding=1, bias=None, kernel_size2=1, padding2=0,stride=1,dilation=1):
        super(DeformRouting, self).__init__()
        self.grid_gen = nn.Conv2d(channels_in, kernel_size ** 2 * 2, kernel_size=kernel_size2, stride=stride,
                                  padding=padding2)
        self.grid_gen.weight.data.fill_(0)
        self.grid_gen.bias.data.fill_(0)


        # channels_in -> channels_in* channels_out*k^2
        self.weight_gen = nn.Conv2d(channels_in, (kernel_size ** 2) * channels_in * channels_out,
                                    kernel_size=kernel_size2, stride=stride,
                                    padding=padding2, bias=True,groups=channels_in)

        fan_in = (kernel_size ** 2) * channels_in
        fan_out = channels_out
        self.weight_gen.weight.data.fill_(0)
        self.weight_gen.bias.data.normal_(0, math.sqrt(2. / (fan_in + fan_out)))


        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.dilation=dilation

    def forward(self, inputs, offset=None, weight=None):
        N, C, H, W = inputs.shape
        kH = self.kernel_size
        kW = self.kernel_size

        if offset is None:
            offset = self.grid_gen(inputs)

        if weight is None:
            weight = self.weight_gen(inputs)

        _,_,H,W=weight.shape
        weight = weight.permute(0, 2, 3, 1).contiguous().view(N, H, W, self.channels_out, self.channels_in * kH * kW)

        # calculate the coordinates of the sampling grid
        base_grid = inputs.new_zeros(1, H, W, 1, 1, 2,requires_grad=False)
        linear_points = torch.from_numpy(np.linspace(-1., 1., W)).type_as(base_grid).to(inputs.device)
        base_grid[..., 0] = torch.ger(inputs.new_ones(H,requires_grad=False), linear_points).view(1, H, W, 1, 1)
        linear_points = torch.from_numpy(np.linspace(-1., 1., H)).type_as(base_grid).to(inputs.device)
        base_grid[..., 1] = torch.ger(linear_points, inputs.new_ones(W,requires_grad=False)).view(1, H, W, 1, 1)

        # calculate the offset for each kernel entry
        kernel_offset = inputs.new_zeros(1, 1, 1, kH, kW, 2 ,requires_grad=False)
        linear_points = torch.from_numpy(np.linspace(-(kW - 1)/ (W - 1)  * self.dilation, (kW - 1) / (W - 1)* self.dilation, kW)).type_as(
            kernel_offset).to(inputs.device)
        kernel_offset[..., 0] = torch.ger(inputs.new_ones(kH,requires_grad=False), linear_points).view(1, 1, 1, kH, kW)
        linear_points = torch.from_numpy(np.linspace(-(kH - 1) / (H - 1) * self.dilation, (kH - 1) / (H - 1)* self.dilation, kH)).type_as(
            kernel_offset).to(inputs.device)
        kernel_offset[..., 1] = torch.ger(linear_points, inputs.new_ones(kW,requires_grad=False)).view(1, 1, 1, kH, kW)

        # (1,H,W, kH, kW,2)
        base_grid = Variable(base_grid + kernel_offset, requires_grad=False)

        # (N,H,W, kH, kW,2)
        offset = offset.permute(0, 2, 3, 1).contiguous().view(N, H, W, kH, kW, 2)

        grid = offset + base_grid

        # (N,2,W,kW,H*kH)
        grid = grid.permute(0, 5, 2, 4, 1, 3).contiguous().view(N, 2, W, kW, H * kH)
        # (N, 2, H*kH, W, kW)  -> (N, H*kH, W*kW, 2)
        grid = grid.permute(0, 1, 4, 2, 3).contiguous().view(N, 2, H * kH, W * kW).permute(0, 2, 3, 1)

        input_wrapped = F.grid_sample(inputs, grid)  # ,padding_mode='border'


        input_wrapped = input_wrapped.view(N, C, H * kH, W, kW).contiguous()
        input_wrapped = input_wrapped.permute(0, 1, 3, 4, 2).contiguous().view(N, C, W, kW, H, kH).contiguous()
        # N, H, W, C, kH, kW
        input_wrapped = input_wrapped.permute(0, 4, 2, 1, 5, 3).contiguous().view(N, H, W, -1, 1)

        # deformable routing
        output = torch.matmul(weight, input_wrapped).squeeze(-1).permute(0, 3, 1, 2).contiguous()

        return output


class DeformConvTranspose2D(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size=2, padding=0, stride=2, bias=None):
        super(DeformConvTranspose2D, self).__init__()
        self.grid_gen = nn.Conv2d(channels_in, 2, kernel_size=1, stride=1, padding=0)
        self.grid_gen.weight.data.fill_(0)
        self.grid_gen.bias.data.fill_(0)
        self.deform_convt = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride,
                                               padding=padding, bias=bias)
        self.kernel_size = 1

    def forward(self, inputs, output_size=None, offset=None):

        N, C, H, W = inputs.shape
        kH = self.kernel_size
        kW = self.kernel_size

        if offset is None:
            offset = self.grid_gen(inputs)

        # calculate the coordinates of the sampling grid
        base_grid = inputs.new_zeros(1, H, W, 1, 1, 2,requires_grad=False)
        linear_points = torch.from_numpy(np.linspace(-1., 1., W)).type_as(base_grid).to(inputs.device)
        base_grid[..., 0] = torch.ger(inputs.new_ones(H,requires_grad=False), linear_points).view(1, H, W, 1, 1)
        linear_points = torch.from_numpy(np.linspace(-1., 1., H)).type_as(base_grid).to(inputs.device)
        base_grid[..., 1] = torch.ger(linear_points, inputs.new_ones(W,requires_grad=False)).view(1, H, W, 1, 1)

        # calculate the offset for each kernel entry
        kernel_offset = inputs.new_zeros(1, 1, 1, kH, kW, 2,requires_grad=False)
        linear_points = torch.from_numpy(np.linspace(-(kW - 1) / (W - 1), (kW - 1) / (W - 1), kW)).type_as(
            kernel_offset).to(inputs.device)
        kernel_offset[..., 0] = torch.ger(inputs.new_ones(kH,requires_grad=False), linear_points).view(1, 1, 1, kH, kW)
        linear_points = torch.from_numpy(np.linspace(-(kH - 1) / (H - 1), (kH - 1) / (H - 1), kH)).type_as(
            kernel_offset).to(inputs.device)
        kernel_offset[..., 1] = torch.ger(linear_points, inputs.new_ones(kW,requires_grad=False)).view(1, 1, 1, kH, kW)

        # (1,H,W, kH, kW,2)
        base_grid = Variable(base_grid + kernel_offset, requires_grad=False)
        offset = offset.permute(0, 2, 3, 1).contiguous().view(N, H, W, kH, kW, 2)

        grid = offset + base_grid

        # double check:
        # (N,2,W,kW,H*kH)
        grid = grid.permute(0, 5, 2, 4, 1, 3).contiguous().view(N, 2, W, kW, H * kH)
        # (N,H*kH,W*kW,2)
        grid = grid.permute(0, 1, 4, 2, 3).contiguous().view(N, 2, H * kH, W * kW).permute(0, 2, 3, 1)

        input_wrapped = F.grid_sample(inputs, grid)  # ,padding_mode='border'

        if output_size is not None:
            output = self.deform_convt(input_wrapped, output_size=output_size)
        else:
            output = self.deform_convt(input_wrapped)

        return output
