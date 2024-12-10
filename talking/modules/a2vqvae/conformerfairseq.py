#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : Torchaudio
Date               : 2023-06-20 06:33
Last Modified By   : Tianyu He (xxxx@microsoft.com)
Last Modified Date : 2023-06-28 18:01
Description        : https://github.com/pytorch/audio/blob/v0.13.1/torchaudio/models/conformer.py
-------- 
Copyright (c) 2023 Microsoft Corporation.
'''

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from fairseq.modules import (
    ESPNETMultiHeadedAttention,
    LayerNorm,
    MultiheadAttention,
    RelPositionMultiHeadedAttention,
    RotaryPositionMultiHeadedAttention,
    PositionalEmbedding,
    RelPositionalEncoding,
)
from fairseq.utils import get_activation_fn


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, pe_scale):
        super().__init__()
        self.dim = dim
        self.pe_scale = pe_scale

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * self.pe_scale
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.to(dtype=dtype)


class PositionalEncoding(torch.nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ConvolutionModule(torch.nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(
        self,
        embed_dim,
        channels,
        depthwise_kernel_size,
        dropout,
        activation_fn="swish",
        bias=False,
        export=False,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            channels: Number of channels in depthwise conv layers
            depthwise_kernel_size: Depthwise conv layer kernel size
            dropout: dropout value
            activation_fn: Activation function to use after depthwise convolution kernel
            bias: If bias should be added to conv layers
            export: If layernorm should be exported to jit
        """
        super(ConvolutionModule, self).__init__()
        assert (
            depthwise_kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        self.layer_norm = LayerNorm(embed_dim, export=export)
        self.pointwise_conv1 = torch.nn.Conv1d(
            embed_dim,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(
            channels,
            channels,
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.batch_norm = torch.nn.BatchNorm1d(channels)
        self.activation = get_activation_fn(activation_fn)(channels)
        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input of shape B X T X C
        Returns:
          Tensor of shape B X T X C
        """
        x = self.layer_norm(x)
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = self.glu(x)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class FeedForwardModule(torch.nn.Module):
    """Positionwise feed forward layer used in conformer"""

    def __init__(
        self,
        input_feat,
        hidden_units,
        dropout1,
        dropout2,
        activation_fn="swish",
        bias=True,
    ):
        """
        Args:
            input_feat: Input feature dimension
            hidden_units: Hidden unit dimension
            dropout1: dropout value for layer1
            dropout2: dropout value for layer2
            activation_fn: Name of activation function
            bias: If linear layers should have bias
        """

        super(FeedForwardModule, self).__init__()
        self.layer_norm = LayerNorm(input_feat)
        self.w_1 = torch.nn.Linear(input_feat, hidden_units, bias=bias)
        self.w_2 = torch.nn.Linear(hidden_units, input_feat, bias=bias)
        self.dropout1 = torch.nn.Dropout(dropout1)
        self.dropout2 = torch.nn.Dropout(dropout2)
        self.activation = get_activation_fn(activation_fn)(hidden_units)

    def forward(self, x):
        """
        Args:
            x: Input Tensor of shape  T X B X C
        Returns:
            Tensor of shape T X B X C
        """
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        return self.dropout2(x)


class ConformerEncoderLayer(torch.nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100. We currently don't support relative positional encoding in MHA"""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        dropout,
        depthwise_conv_kernel_size=31,
        activation_fn="swish",
        attn_type=None,
        pos_enc_type="abs",
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            ffn_embed_dim: FFN layer dimension
            attention_heads: Number of attention heads in MHA
            dropout: dropout value
            depthwise_conv_kernel_size: Size of kernel in depthwise conv layer in convolution module
            activation_fn: Activation function name to use in convulation block and feed forward block
            attn_type: MHA implementation from ESPNET vs fairseq
            pos_enc_type: Positional encoding type - abs, rope, rel_pos
        """
        self.pos_enc_type = pos_enc_type
        super(ConformerEncoderLayer, self).__init__()

        self.ffn1 = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
        )

        self.self_attn_layer_norm = LayerNorm(embed_dim, export=False)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        if attn_type == "espnet":
            if self.pos_enc_type == "rel_pos":
                self.self_attn = RelPositionMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                )
            elif self.pos_enc_type == "rope":
                # precision is not used actually
                self.self_attn = RotaryPositionMultiHeadedAttention(
                    embed_dim, attention_heads, dropout=dropout, precision="fp16"
                )
            elif self.pos_enc_type == "abs":
                self.self_attn = ESPNETMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                )
            else:
                raise Exception(f"Unsupported attention type {self.pos_enc_type}")
        else:
            # Default to fairseq MHA
            self.self_attn = MultiheadAttention(
                embed_dim,
                attention_heads,
                dropout=dropout,
            )

        self.conv_module = ConvolutionModule(
            embed_dim=embed_dim,
            channels=embed_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            activation_fn=activation_fn,
        )

        self.ffn2 = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
            activation_fn=activation_fn,
        )
        self.final_layer_norm = LayerNorm(embed_dim, export=False)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[torch.Tensor],
        position_emb: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: Tensor of shape T X B X C
            encoder_padding_mask: Optional mask tensor
            positions:
        Returns:
            Tensor of shape T X B X C
        """
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual
        residual = x
        x = self.self_attn_layer_norm(x)
        if self.pos_enc_type == "rel_pos":
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                pos_emb=position_emb,
                need_weights=False,
            )
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
            )
        x = self.self_attn_dropout(x)
        x = x + residual

        residual = x
        # TBC to BTC
        x = x.transpose(0, 1)
        x = self.conv_module(x)
        # BTC to TBC
        x = x.transpose(0, 1)
        x = residual + x

        residual = x
        x = self.ffn2(x)

        layer_result = x

        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x, (attn, layer_result)


class DiffConformerEncoder(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.
    Args:
        d_model (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    Examples:
        >>> conformer = Conformer(
        >>>     d_model=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), d_model)  # (batch, num_frames, d_model)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        in_dims,
        d_model,
        out_dims,
        num_heads,
        ffn_dim,
        cond_mode,
        cond_dim,
        num_layers,
        depthwise_conv_kernel_size,
        pos_enc_type="abs",
        max_source_positions=1024,
        dropout=0.0,
        use_group_norm=False,
    ):
        super().__init__()
        # for input dim projection
        self.input_projection = nn.Linear(in_dims, d_model)

        # diffusion timestep embedding
        self.diffusion_embedding = SinusoidalPosEmb(d_model, pe_scale=1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            Mish(),
            nn.Linear(d_model * 4, d_model)
        )

        # positional embedding
        self.pos_enc_type = pos_enc_type
        if self.pos_enc_type == "rel_pos":
            self.embed_positions = RelPositionalEncoding(max_source_positions, d_model)
        elif self.pos_enc_type == "rope":
            self.embed_positions = None
        else:  # Use absolute positional embedding
            self.pos_enc_type = "abs"
            self.embed_positions = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=max_source_positions)

        # conformer layers
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    embed_dim=d_model,
                    ffn_embed_dim=ffn_dim,
                    attention_heads=num_heads,
                    dropout=dropout,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    attn_type='fairseq',
                    pos_enc_type=pos_enc_type
                )
                for _ in range(num_layers)
            ]
        )

        self.out_proj = torch.nn.Linear(d_model, out_dims) if out_dims > 0 else torch.nn.Identity()

    def forward(self, input, diffusion_step, cond1=None, cond2=None, masks=None):
        r"""
        Concatenates the condition to each layer input of the conformer.
        Args:
            input (torch.Tensor): with shape `(B, T, d_model)`.
            masks (torch.Tensor): with shape `(B, T, 1)`. 1 indicates positive.
        Returns:
            torch.Tensor
                output frames, with shape `(B, T, d_model)`
        """
        # encoder_padding_mask_old = _lengths_to_padding_mask(lengths)  # (bs, T), False indicates positive
        # input: (bs, T, d)
        # cond: (bs, T, d)
        # prepare condition
        x = self.input_projection(input)
        x = F.relu(x)

        if cond2 is None:
            cond = cond1
        else:
            cond = cond1 + cond2
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        x = x.transpose(0, 1)

        if self.add_pos_emb:
            # x = x + sinusoidal_positional_encoding
            x = self.pos_enc(x) # (T, bs, d)

        position_emb = self.embed_positions(x) if self.use_rel_attn else None

        for layer in self.conformer_layers:
            x = layer(x, cond=cond.transpose(0, 1), diffusion_step=diffusion_step, position_emb=position_emb)
        x = self.out_proj(x)
        return x.transpose(0, 1)