import torch
from torch import nn
from .transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerDecoder,
)

from .self_attention_pooling import SelfAttentionPooling

def _reset_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
class StyleEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        pos_embed_len=250,
        input_dim=768,
        aggregate_method="average",
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        _reset_parameters(self.encoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        self.increase_embed_dim = nn.Linear(input_dim, d_model)

        self.aggregate_method = None
        if aggregate_method == "self_attention_pooling":
            self.aggregate_method = SelfAttentionPooling(d_model)
        elif aggregate_method == "average":
            pass
        else:
            raise ValueError(f"Invalid aggregate method {aggregate_method}")

    def forward(self, x, pad_mask=None):
        """

        Args:
            x (_type_): (B, num_frames(L), C_exp)
            pad_mask: (B, num_frames)

        Returns:
            style_code: (B, C_model)
        """
        x = self.increase_embed_dim(x)
        # (B, L, C)
        x = x.permute(1, 0, 2)
        # (L, B, C)

        pos = self.pos_embed(x.shape[0])
        pos = pos.permute(1, 0, 2)
        # (L, 1, C)

        style = self.encoder(x, pos=pos, src_key_padding_mask=pad_mask)
        # (L, B, C)

        if self.aggregate_method is not None:
            permute_style = style.permute(1, 0, 2)
            # (B, L, C)
            style_code = self.aggregate_method(permute_style, pad_mask)
            return style_code

        if pad_mask is None:
            style = style.permute(1, 2, 0)
            # (B, C, L)
            style_code = style.mean(2)
            # (B, C)
        else:
            permute_style = style.permute(1, 0, 2)
            # (B, L, C)
            permute_style[pad_mask] = 0
            sum_style_code = permute_style.sum(dim=1)
            # (B, C)
            valid_token_num = (~pad_mask).sum(dim=1).unsqueeze(-1)
            # (B, 1)
            style_code = sum_style_code / valid_token_num
            # (B, C)

        return style_code
