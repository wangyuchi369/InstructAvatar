#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : Torchaudio
Date               : 2023-06-20 06:33
Last Modified By   : Tianyu He (xxxx@microsoft.com)
Last Modified Date : 2023-06-29 23:05
Description        : https://github.com/pytorch/audio/blob/v0.13.1/torchaudio/models/conformer.py
-------- 
Copyright (c) 2023 Microsoft Corporation.
'''

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from talking.modules.a2diffae.conformerfairseq import RelPositionMultiHeadedAttention, RelPositionalEncoding, PositionalEmbedding
from fairseq.modules import MultiheadAttention
from talking.modules.a2vqvae.attention import xMultiheadAttention


__all__ = ["Conformer", "CondConformer"]


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


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


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py#L448
    # original num_embeddings: The number of diffusion steps used during training.

    def __init__(self, embedding_dim, cond_dim):
        super().__init__()
        # self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.emb = nn.Linear(cond_dim, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, cond=None):
        cond = x if cond == None else cond
        emb = self.linear(self.silu(self.emb(cond)))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(nn.Module):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).
    """
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py#L467
    # original num_embeddings: The number of diffusion steps used during training.

    def __init__(self, embedding_dim):
        super().__init__()
        # self.emb = nn.Embedding(num_embeddings, embedding_dim)
        # self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        self.emb = nn.Linear(embedding_dim, embedding_dim)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, cond=None):
        cond = x if cond == None else cond
        emb = self.linear(self.silu(self.emb(cond)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class _AdaConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        # self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.layer_norm = AdaLayerNorm(input_dim, cond_dim)
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

    def forward(self, input, cond):
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.
        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input, cond)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class _AdaFeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.
    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim, hidden_dim, cond_dim, dropout=0.0):
        super().__init__()
        self.adaln = AdaLayerNorm(input_dim, cond_dim)
        self.sequential = torch.nn.Sequential(
            # torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input, cond):
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.
        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        input = self.adaln(input, cond)
        return self.sequential(input)


class ConformerLayerconcateLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.
    NOTE: support cond=None, which will degrade to the original conformer.
    """

    def __init__(
        self,
        input_dim,
        ffn_dim,
        cond_dim,
        num_attention_heads,
        depthwise_conv_kernel_size,
        use_rel_attn=False,
        dropout=0.0,
        use_group_norm=False,
        convolution_first=False,
    ):
        super().__init__()
        # condition projection
        self.diffusion_projection = nn.Linear(input_dim, input_dim)
        self.cond_projection = nn.Linear(input_dim, input_dim)

        self.in_proj = torch.nn.Linear(input_dim + cond_dim, input_dim)

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)

        self.use_rel_attn = use_rel_attn
        if use_rel_attn:
            self.self_attn = RelPositionMultiHeadedAttention(input_dim, num_attention_heads, dropout)
        else:
            # self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
            self.self_attn = xMultiheadAttention(input_dim, num_attention_heads, dropout=dropout)

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

    def forward(self, input, cond=None, diffusion_step=None, key_padding_mask=None, position_emb=None):
        r"""
        NOTE: support cond=None, which will degrade to the original conformer.
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.
        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        cond = self.diffusion_projection(diffusion_step).unsqueeze(0) + self.cond_projection(cond)
        
        if cond is not None:
            input = self.in_proj(torch.cat([input, cond], dim=-1))

        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x)

        # key_padding_mask: (bs, T), True will be ignored
        residual = x
        x = self.self_attn_layer_norm(x)
        # https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/conformer_layer.py#L250
        if self.use_rel_attn:
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
                pos_emb=position_emb,
                need_weights=False,
            )
        else:
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


class ConformerAdaLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.
    """

    def __init__(
        self,
        input_dim,
        ffn_dim,
        cond_dim,
        num_attention_heads,
        depthwise_conv_kernel_size,
        use_rel_attn=False,
        dropout=0.0,
        use_group_norm=False,
        convolution_first=False,
    ):
        super().__init__()
        # condition projection
        self.diffusion_projection = nn.Linear(input_dim, input_dim)
        self.cond_projection = nn.Linear(input_dim, input_dim)

        self.ffn1 = _AdaFeedForwardModule(input_dim, ffn_dim, cond_dim, dropout=dropout)

        # self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn_layer_norm = AdaLayerNorm(input_dim, cond_dim)

        self.use_rel_attn = use_rel_attn
        if use_rel_attn:
            self.self_attn = RelPositionMultiHeadedAttention(input_dim, num_attention_heads, dropout)
        else:
            # self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
            # xformers_att_config = '{"name": "scaled_dot_product"}'
            # xformers_att_config = '{"name": "linformer", "seq_len": 250}'
            # self.self_attn = MultiheadAttention(input_dim, num_attention_heads, dropout=dropout, xformers_att_config=xformers_att_config)
            self.self_attn = xMultiheadAttention(input_dim, num_attention_heads, dropout=dropout)

        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _AdaConvolutionModule(
            input_dim=input_dim,
            cond_dim=cond_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _AdaFeedForwardModule(input_dim, ffn_dim, cond_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input, cond):
        residual = input
        input = input.transpose(0, 1)
        cond = cond.transpose(0, 1)
        input = self.conv_module(input, cond)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input, cond=None, diffusion_step=None, key_padding_mask=None, position_emb=None):
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, bs, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.
        Returns:
            torch.Tensor: output, with shape `(T, bs, D)`.
        """
        input = input + self.diffusion_projection(diffusion_step).unsqueeze(0) + self.cond_projection(cond)

        residual = input
        x = self.ffn1(input, cond)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x, cond)

        residual = x
        x = self.self_attn_layer_norm(x, cond)
        # https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/conformer_layer.py#L250
        if self.use_rel_attn:
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
                pos_emb=position_emb,
                need_weights=False,
            )
        else:
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
            # cond: (T, bs, d)
            x = self._apply_convolution(x, cond)

        residual = x
        x = self.ffn2(x, cond)
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
        cond_mode,  # ada, layerconcat
        in_dims, 
        d_model, 
        out_dims, 
        num_heads,
        ffn_dim,
        cond_dim,
        num_layers,
        depthwise_conv_kernel_size,
        use_rel_attn=False,
        max_source_positions=1024,
        dropout=0.0,
        use_group_norm=False,
        convolution_first=False,
        add_pos_emb=True
    ):
        super().__init__()
        self.input_projection = nn.Linear(in_dims, d_model)
        self.diffusion_embedding = SinusoidalPosEmb(d_model, pe_scale=1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            Mish(),
            nn.Linear(d_model * 4, d_model)
        )

        self.add_pos_emb = add_pos_emb
        if add_pos_emb:
            self.pos_enc = PositionalEncoding(d_model)

        self.use_rel_attn = use_rel_attn
        if use_rel_attn:
            self.embed_positions = RelPositionalEncoding(max_source_positions, d_model)

        if cond_mode == 'ada':
            Layer = ConformerAdaLayer
        elif cond_mode == 'layerconcat':
            Layer = ConformerLayerconcateLayer
        else:
            raise ValueError(f'[CondConformer] cond ({cond_mode}) must be: ada, layerconcat')

        self.conformer_layers = torch.nn.ModuleList(
            [
                Layer(d_model,
                      ffn_dim,
                      cond_dim,
                      num_heads,
                      depthwise_conv_kernel_size,
                      use_rel_attn=use_rel_attn,
                      dropout=dropout,
                      use_group_norm=use_group_norm,
                      convolution_first=convolution_first,
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
        key_padding_mask = ~masks[..., 0].to(torch.bool) if masks is not None else None

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
            x = layer(x, cond=cond.transpose(0, 1), diffusion_step=diffusion_step, key_padding_mask=key_padding_mask, position_emb=position_emb)
        x = self.out_proj(x)
        return x.transpose(0, 1)