import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint as checkpoint
# from ldm.modules.diffusionmodules.util import checkpoint


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


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


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self,
                 encoder_hidden,
                 residual_channels,
                 dilation,
                 gradient_checkpointing):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    # https://github.com/lucidrains/DALLE2-pytorch/issues/225
    def forward(self, x, conditioner, diffusion_step):
        if self.gradient_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward, x, conditioner, diffusion_step)
        else:
            return self._forward(x, conditioner, diffusion_step)

    # def forward(self, x, conditioner, diffusion_step):
    #     return checkpoint(self._forward, (x, conditioner, diffusion_step), self.parameters(), self.gradient_checkpointing)

    def _forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class WavNet(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dims,
                 residual_layers=20,
                 residual_channels=512,
                 dilation_cycle_length=2,
                 pe_scale=1,
                 gradient_checkpointing=False):
        super().__init__()
        self.input_projection = Conv1d(in_dims, residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(residual_channels, pe_scale=pe_scale)
        dim = residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels, residual_channels, 2 ** (i % dilation_cycle_length), gradient_checkpointing)
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, out_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, input, diffusion_step, cond1, cond2=None, masks=None):
        """
        input: (B, d, T)
        diffusion_step: (B)
        cond: (B, d, T)
        """
        # cond1: audio
        # cond2: keyla
        input, cond1, cond2, masks = [x.transpose(1, 2) for x in [input, cond1, cond2, masks]]

        if cond2 is None:
            cond = cond1
        else:
            cond = cond1 + cond2

        x = self.input_projection(input)

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x.transpose(1, 2)
