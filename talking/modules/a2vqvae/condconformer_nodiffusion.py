#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : Torchaudio
Date               : 2023-06-20 06:33
Last Modified By   : Tianyu He (xxxx@microsoft.com)
Last Modified Date : 2023-07-11 03:42
Description        : https://github.com/pytorch/audio/blob/v0.13.1/torchaudio/models/conformer.py
-------- 
Copyright (c) 2023 Microsoft Corporation.
'''

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


__all__ = ["PosConformer", "CondConformer"]


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class FiLM(nn.Module):
    def __init__(self, in_channels, condition_channels):
        super().__init__()
        self.gain = nn.Linear(condition_channels, in_channels)
        self.bias = nn.Linear(condition_channels, in_channels)

        nn.init.xavier_uniform(self.gain.weight)
        nn.init.constant(self.gain.bias, 1)

        nn.init.xavier_uniform(self.bias.weight)
        nn.init.constant(self.bias.bias, 0)

    def forward(self, x, condition):
        gain = self.gain(condition)
        bias = self.bias(condition)
        if gain.dim() == 2:
            gain = gain.unsqueeze(-1)
        if bias.ndim == 2:
            bias = bias.unsqueeze(-1)
        return x * gain + bias


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


class _ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.
    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else torch.nn.BatchNorm1d(num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.
        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class _FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.
    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.
        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)


class ConformerLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.
    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.
        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x


class Conformer(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.
    Args:
        input_dim (int): input dimension.
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
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        cond_dim: int = 0,  # for compatibility
    ):
        super().__init__()

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input, masks, cond=None):
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        # encoder_padding_mask_old = _lengths_to_padding_mask(lengths)  # (bs, T), False indicates positive
        # input: (bs, T, d)
        # masks: (bs, T, 1), 1 indicates positive
        encoder_padding_mask = ~masks[..., 0].to(torch.bool)

        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return x.transpose(0, 1)


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


class PosConformer(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.
    Args:
        input_dim (int): input dimension.
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
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        add_pos_emb: bool = True,
        output_dim: int = 0,  # vary the output dim
        cond_dim: int = 0,  # for compatibility
    ):
        super().__init__()

        self.add_pos_emb = add_pos_emb
        if add_pos_emb:
            self.pos_enc = PositionalEncoding(input_dim)

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = torch.nn.Linear(input_dim, output_dim) if output_dim > 0 else torch.nn.Identity()

    def forward(self, input, masks, cond=None):
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        # encoder_padding_mask_old = _lengths_to_padding_mask(lengths)  # (bs, T), False indicates positive
        # input: (bs, T, d)
        # masks: (bs, T, 1), 1 indicates positive
        encoder_padding_mask = ~masks[..., 0].to(torch.bool) if masks is not None else None

        x = input.transpose(0, 1)
        if self.add_pos_emb:
            x = self.pos_enc(x)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        x = self.out_proj(x)
        return x.transpose(0, 1)


class CondConformerLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.
    """

    def __init__(
        self,
        input_dim,
        ffn_dim,
        cond_keyinfo_dim,
        num_attention_heads,
        depthwise_conv_kernel_size,
        dropout=0.0,
        use_group_norm=False,
        convolution_first=False,
    ):
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.cross_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.cross_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout,
                                                      kdim=cond_keyinfo_dim, vdim=cond_keyinfo_dim)
        self.cross_attn_dropout = torch.nn.Dropout(dropout)
        self.film = FiLM(input_dim, input_dim)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input, cond_keyinfo=None, key_padding_mask=None):
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, bs, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.
        Returns:
            torch.Tensor: output, with shape `(T, bs, D)`.
        """

        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x)

        # self attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)

        # cond_keyinfo = cond_keyinfo[0].unsqueeze(0)
        # add keyinfo condition by cross attention
        residual = x
        x = self.cross_attn_layer_norm(x)
        x, _ = self.cross_attn(
            query=x,
            key=cond_keyinfo,
            value=cond_keyinfo,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.cross_attn_dropout(x)
        x = x + residual  # (T, bs, D)
        # FiLM
        x = self.film(residual, x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x



class CondConformer(torch.nn.Module):
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
        cond_keyinfo_dim,
        num_layers,
        depthwise_conv_kernel_size,
        dropout=0.0,
        use_group_norm=False,
        convolution_first=False,
        add_pos_emb=True
    ):
        super().__init__()
        self.input_projection = nn.Linear(in_dims, d_model)

        self.add_pos_emb = add_pos_emb
        if add_pos_emb:
            self.pos_enc = PositionalEncoding(d_model)

        self.conformer_layers = torch.nn.ModuleList(
            [
                CondConformerLayer(d_model,
                                   ffn_dim,
                                   cond_keyinfo_dim,
                                   num_heads,
                                   depthwise_conv_kernel_size,
                                   dropout=dropout,
                                   use_group_norm=use_group_norm,
                                   convolution_first=convolution_first)
                for _ in range(num_layers)
            ]
        )
        self.out_proj = torch.nn.Linear(d_model, out_dims) if out_dims > 0 else torch.nn.Identity()

    def forward(self, input, cond1=None, cond2=None, masks=None):
        r"""
        Concatenates the condition to each layer input of the conformer.
        Args:
            input (torch.Tensor): with shape `(B, T, d_model)`.
            cond1 (torch.Tensor): audio.
            cond2 (torch.Tensor): keyframe info.
            masks (torch.Tensor): with shape `(B, T, 1)`. 1 indicates positive.
        Returns:
            torch.Tensor
                output frames, with shape `(B, T, d_model)`
        """
        # encoder_padding_mask_old = _lengths_to_padding_mask(lengths)  # (bs, T), False indicates positive
        # input: (bs, T, d)
        # cond: (bs, T, d)
        # prepare condition
        # key_padding_mask = ~masks[..., 0].to(torch.bool) if masks is not None else None

        x = self.input_projection(input)
        x = F.relu(x)

        x = x.transpose(0, 1)

        if self.add_pos_emb:
            # x = x + sinusoidal_positional_encoding
            x = self.pos_enc(x) # (T, bs, d)

        for layer in self.conformer_layers:
            x = layer(x, cond_keyinfo=cond2.transpose(0, 1), key_padding_mask=None)
        x = self.out_proj(x)
        return x.transpose(0, 1)
    