#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : Tianyu He (xxxx@microsoft.com)
Date               : 2023-01-17 00:09
Last Modified By   : Tianyu He (xxxx@microsoft.com)
Last Modified Date : 2023-04-15 21:41
Description        : 
-------- 
Copyright (c) 2023 Microsoft Corporation.
'''


import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TalkingMLPEncoder', 'TalkingMLPEncoderLayer', '_get_activation_fn', '_get_clones', 'MLPMixerEncoder']


class TalkingMLPEncoder(nn.Module):
    """
    reference:
    https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
    """    
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim,
                 num_layers,
                 dropout=0.1):
        """
        Parameters
        ----------
        in_dim : int
            input dim
        out_dim : int
            should be same as embed_dim
        hidden_dim : int
            hidden dim between layers
        num_layers : int
            number of layers
        """                 
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        encoder_layer = TalkingMLPEncoderLayer(hidden_dim, hidden_dim,
                                        activation='relu', dropout=dropout)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.in_proj(x)
        x = F.relu(x)

        for mod in self.layers:
            x = mod(x)
        return self.out_proj(x)


class TalkingMLPEncoderLayer(nn.Module):
    """
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 activation='relu',
                 dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # feed forward block
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x):
        x = self.activation(self.linear(x))
        return self.dropout(x)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLPMixerEncoder(nn.Module):
    """reference: https://github.com/lucidrains/mlp-mixer-pytorch
    """    
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        return