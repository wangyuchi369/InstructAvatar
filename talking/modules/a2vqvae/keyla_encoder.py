#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : Tianyu He (xxxx@microsoft.com)
Date               : 2023-05-30 19:06
Last Modified By   : Tianyu He (xxxx@microsoft.com)
Last Modified Date : 2023-05-30 20:02
Description        : 
-------- 
Copyright (c) 2023 Microsoft Corporation.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class KeylaEncoder(nn.Module):
    def __init__(self,
                 in_shape,
                 out_dim,
                 activation='relu',
                 dropout=0.1):
        super().__init__()

        assert len(in_shape) == 3, "[KeylaEncoder] in_shape should be (C, H, W), got {}".format(in_shape)

        self.in_shape = in_shape
        in_dim = in_shape[0]

        self.in_conv = torch.nn.Conv2d(in_dim,
                                       in_dim * 2,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        # self.in_block = ResnetBlock(in_dim * 2, in_dim * 2, True, dropout)
        self.in_down = Downsample(in_dim * 2, True)

        self.mid_block = ResnetBlock(in_dim * 2, in_dim * 2, True, dropout)
        self.mid_down = Downsample(in_dim * 2, True)

        self.out_norm = Normalize(in_dim * 2)
        self.out_block = ResnetBlock(in_dim * 2, in_dim, True, dropout)

        self.out_proj1 = nn.Linear(int(in_dim * in_shape[1] * in_shape[2] / 16), out_dim)
        self.out_drop = nn.Dropout(dropout)
        self.out_proj2 = nn.Linear(out_dim, out_dim)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation


    def forward(self, x):
        bs, T, d = x.shape
        x = x.view(bs*T, *self.in_shape).contiguous()
        x = self.in_conv(x)
        # x = self.in_block(x)
        x = nonlinearity(x)
        x = self.in_down(x)
        x = self.mid_block(x)
        x = self.mid_down(x)

        x = self.out_norm(x)
        x = nonlinearity(x)
        x = self.out_block(x)

        x = x.view(bs, T, -1)
        x = self.activation(self.out_proj1(x))
        x = self.out_drop(x)
        x = self.out_proj2(x)

        return x