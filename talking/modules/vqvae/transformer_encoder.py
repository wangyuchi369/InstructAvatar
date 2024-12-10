import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat

from ldm.modules.encoders.modules import AbstractEncoder
from ldm.modules.x_transformer import Encoder, AttentionLayers, default, AbsolutePositionalEmbedding, always, exists


class TransformerEncoder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, in_dim, embed_dim, n_layer, max_seq_len=77):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, embed_dim) if in_dim != embed_dim else nn.Identity()
        self.transformer = TransformerWrapper(max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=embed_dim, depth=n_layer))

    def forward(self, x):
        x = self.in_proj(x)
        z = self.transformer(x, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class TransformerWrapper(nn.Module):
    """
    from ldm.modules.x_transformer.TransformerWrapper
    """
    def __init__(
            self,
            *,
            max_seq_len,
            attn_layers,
            emb_dim=None,
            max_mem_len=0.,
            emb_dropout=0.,
            num_memory_tokens=None,
            tie_embedding=False,
            use_pos_emb=True
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len

        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len) if (
                    use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

            # let funnel encoder know number of memory tokens, if specified
            if hasattr(attn_layers, 'num_memory_tokens'):
                attn_layers.num_memory_tokens = num_memory_tokens

    def forward(
            self,
            x,
            return_embeddings=False,
            mask=None,
            return_mems=False,
            return_attn=False,
            mems=None,
            **kwargs
    ):
        b, num_mem = x.shape[0], self.num_memory_tokens
        x += self.pos_emb(x)
        x = self.emb_dropout(x)
        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, 'n d -> b n d', b=b)
            x = torch.cat((mem, x), dim=1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)

        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = self.to_logits(x) if not return_embeddings else x

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))
            return out, new_mems

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out