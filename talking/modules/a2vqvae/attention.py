#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : Tianyu He (xxxx@microsoft.com)
Date               : 2023-06-29 13:00
Last Modified By   : Tianyu He (xxxx@microsoft.com)
Last Modified Date : 2023-06-29 21:52
Description        : 
-------- 
Copyright (c) 2023 Microsoft Corporation.
'''


import math
import numpy as np
from einops import rearrange
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter

try:
    import xformers
    import xformers.ops
    device_name = torch.cuda.get_device_name(0)
    # xformer should be compiled from source
    # Pascal (sm60), Volta (sm70), Turing (sm75), Ampere (sm80)
    XFORMERS_IS_AVAILBLE = True
    print(f"[attention] Using XFORMERS with {device_name}.")
except:
    XFORMERS_IS_AVAILBLE = False
    print(f"[attention] No module 'xformers'. Proceeding without it.")


def _mask_for_xformers(mask: torch.Tensor, to_dtype: Optional[torch.dtype] = None):
    """
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/multihead_attention.py
    call to pytorch multihead accepts three mask types:
        - ByteTensor where non-zero means to mask
        - FloatTensor which is an additive mask
        - BoolTensor where True means to mask
    xFormers currently accepts boolean and additive maks. For boolean masks
    the values have opposite meaning. For a BoolTensor True mean to keep the value.
    """
    float_types = [torch.float, torch.float16]
    # If an input mask is a float it is an additive mask. Otherwise it is either uint8 or bool.
    additive = mask.dtype in float_types
    # If to_dype is not specified, keep same dtype as mask.
    to_dtype = mask.dtype if to_dtype is None else to_dtype
    to_additive = to_dtype in float_types

    if additive:
        if to_additive:
            return mask.to(to_dtype)
        mask = mask < 0

    if to_additive:
        # return additive mask
        new_mask = torch.zeros_like(mask, dtype=to_dtype)
        new_mask = new_mask.masked_fill_(mask, -float("inf"))
        return new_mask

    # In xFormers True is value to keep rather than value to mask
    mask = ~mask.to(torch.bool)
    mask = mask.to(to_dtype)
    return mask


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class xMultiheadAttention(nn.Module):
    r"""`Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
    Allows xformers to be used as a drop-in replacement for PyTorch's MultiheadAttention module.
    https://github.dev/pytorch/pytorch/tree/v1.12.1
    https://github.dev/huggingface/transformers/
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, batch_first=False, max_position_embeddings=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.head_dim = embed_dim // num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=max_position_embeddings)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False,
                attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        r"""
        Args:
            query, key, value (torch.Tensor): input, with shape `(T, bs, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer. True will be ignored.
            need_weights (bool): return attention weights.
            attn_mask (torch.Tensor or None): attention mask to use in self attention layer.
            is_causal (bool): whether the attn_mask is causal or not.
        Returns:
            torch.Tensor: output, with shape `(T, bs, D)`.
        """
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        if self.in_proj_bias is not None:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        else:
            b_q = b_k = b_v = None

        if not self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        query = nn.functional.linear(query, w_q, b_q).view(
            query.size(0), query.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        key = nn.functional.linear(key, w_k, b_k).view(
            key.size(0), key.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        value = nn.functional.linear(value, w_v, b_v).view(
            value.size(0), value.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value, seq_len=key.size(2))
        position_ids = torch.arange(key.size(2), device=key.device).unsqueeze(0).view(-1, key.size(2))
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids=position_ids)

        if XFORMERS_IS_AVAILBLE:  # and self.training:
            # xformers: (bs, T, H, D)
            query, key, value = [x.transpose(1, 2) for x in (query, key, value)]
            if is_causal:
                assert attn_mask is None, "[xMultiheadAttention] attn_mask should be None for causal attention."
                attn_mask = xformers.ops.LowerTriangularMask()
            else:
                key_padding_mask = _mask_for_xformers(key_padding_mask, to_dtype=torch.bool)
                # https://github.com/facebookresearch/xformers/blob/main/xformers/components/attention/utils.py#L37
                # Combine the attention mask and key padding mask into a single mask
                # Taken from https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
                # Additive masking not yet supported
                attn_mask = xformers.components.attention.utils.maybe_merge_masks(
                    attn_mask, key_padding_mask, batch_size=query.size(0), src_len=key.size(1), tgt_len=query.size(1), num_heads=self.num_heads)
                attn_mask = xformers.components.attention.utils.bool_mask_to_additive(attn_mask, dtype=query.dtype)
                key_padding_mask = None
                # attn_mask: (bs*H, T, T), -inf for masked positions, 0 for others

            # https://facebookresearch.github.io/xformers/components/ops.html
            # input tensors: (bs, T, H, D)
            # T should be divisible by H
            attn_output = xformers.ops.memory_efficient_attention(
                query, key, value, p=self.dropout, attn_bias=attn_mask)
            attn_output = attn_output.reshape(attn_output.size(0), attn_output.size(1), -1)
        else:
            # PyTorch: (bs, H, T, D)
            bsz, q_len, kv_seq_len = query.size(0), query.size(2), key.size(2)

            attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"[xMultiheadAttention] Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
            if attn_mask is not None:
                if attn_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"[xMultiheadAttention] Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attn_mask.size()}"
                    )
                attn_weights = attn_weights + attn_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_output = torch.matmul(attn_weights, value)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`[xMultiheadAttention] attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.out_proj(attn_output)
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        assert need_weights == False, "[xMultiheadAttention] need_weights is not supported yet."
        attn_output_weights = None

        return attn_output, attn_output_weights
    
    
class ChunkMultiheadAttn(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = False

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

      
        self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
        self.register_parameter('in_proj_weight', None)


        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()