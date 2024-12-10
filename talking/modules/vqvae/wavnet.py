import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import checkpoint


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
                 dilation):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class ResidualBlockCkpt(nn.Module):
    def __init__(self,
                 encoder_hidden,
                 residual_channels,
                 dilation,
                 use_checkpoint):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        return checkpoint(self._forward, (x, conditioner, diffusion_step), self.parameters(), self.use_checkpoint)

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
                 pe_scale=1):
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
            ResidualBlock(residual_channels, residual_channels, 2 ** (i % dilation_cycle_length))
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, out_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, input, diffusion_step, cond1, cond2, masks=None):
        """
        input: (B, d, T)
        diffusion_step: (B)
        cond: (B, d, T)
        """
        # cond1: audio
        # cond2: keyla
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
        return x


class BN3ResidualBlock(nn.Module):
    def __init__(self,
                 encoder_hidden,
                 residual_channels,
                 dilation):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(2 * residual_channels)
        self.norm2 = nn.BatchNorm1d(2 * residual_channels)
        self.norm3 = nn.BatchNorm1d(2 * residual_channels)
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.norm1(self.conditioner_projection(conditioner))
        y = x + diffusion_step

        y = self.norm2(self.dilated_conv(y)) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.norm3(self.output_projection(y))
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class BN3WavNet(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dims,
                 residual_layers=20,
                 residual_channels=512,
                 dilation_cycle_length=2,
                 pe_scale=1):
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
            BN3ResidualBlock(residual_channels, residual_channels, 2 ** (i % dilation_cycle_length))
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, out_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, input, diffusion_step, cond1, cond2, masks=None):
        """
        input: (B, d, T)
        diffusion_step: (B)
        cond: (B, d, T)
        """
        # cond1: audio
        # cond2: keyla
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
        return x


class BN1ResidualBlock(nn.Module):
    def __init__(self,
                 encoder_hidden,
                 residual_channels,
                 dilation):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(2 * residual_channels)
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.norm1(self.output_projection(y))
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class BN1WavNet(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dims,
                 residual_layers=20,
                 residual_channels=512,
                 dilation_cycle_length=2,
                 pe_scale=1):
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
            BN1ResidualBlock(residual_channels, residual_channels, 2 ** (i % dilation_cycle_length))
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, out_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, input, diffusion_step, cond1, cond2, masks=None):
        """
        input: (B, d, T)
        diffusion_step: (B)
        cond: (B, d, T)
        """
        # cond1: audio
        # cond2: keyla
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
        return x


class MaskResidualBlock(nn.Module):
    def __init__(self,
                 encoder_hidden,
                 residual_channels,
                 dilation):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step, masks):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner) * masks
        y = x + diffusion_step

        y = self.dilated_conv(y) * masks + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y) * masks
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class MaskWavNet(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dims,
                 residual_layers=20,
                 residual_channels=512,
                 dilation_cycle_length=2,
                 pe_scale=1):
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
            MaskResidualBlock(residual_channels, residual_channels, 2 ** (i % dilation_cycle_length))
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, out_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, input, diffusion_step, cond1, cond2, masks=None):
        """
        input: (bs, d, T)
        diffusion_step: (bs)
        cond: (bs, d, T)
        masks: (bs, 1, T)
        """
        # cond1: audio
        # cond2: keyla
        cond = cond1 + cond2

        x = self.input_projection(input) * masks

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step, masks)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x) * masks
        x = F.relu(x)
        x = self.output_projection(x) * masks
        return x


class TwoCondResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation, is_audio_add=True):
        super().__init__()
        self.is_audio_add = is_audio_add
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)

        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.add_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.concat_projection = Conv1d(encoder_hidden, residual_channels, 1)
        self.concat_reduce = Conv1d(3 * residual_channels, 2 * residual_channels, 1)

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, cond1, cond2, diffusion_step):
        # cond1: audio
        # cond2: keyla
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)

        if self.is_audio_add:
            add_cond = self.add_projection(cond1)
            concat_cond = self.concat_projection(cond2)
        else:
            add_cond = self.add_projection(cond2)
            concat_cond = self.concat_projection(cond1)

        # x: (bs, 512, T)
        # diffusion_step: (bs, 512, 1)
        y = x + diffusion_step

        y = self.concat_reduce(torch.cat((self.dilated_conv(y), concat_cond), dim=1))
        y = y + add_cond

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class TwoCondWavNet(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dims,
                 is_audio_add=True,
                 residual_layers=20,
                 residual_channels=512,
                 dilation_cycle_length=2,
                 pe_scale=1):
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
            TwoCondResidualBlock(residual_channels, residual_channels, 2 ** (i % dilation_cycle_length), is_audio_add=is_audio_add)
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, out_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, input, diffusion_step, cond1, cond2, masks=None):
        """
        input: (B, d, T)
        diffusion_step: (B)
        cond: (B, d, T)
        """
        # cond1: audio
        # cond2: keyla
        x = self.input_projection(input)

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond1, cond2, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x


class NormResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation, use_norm):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=2 * residual_channels, num_channels=2 * residual_channels) if use_norm else nn.Identity()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.norm(self.dilated_conv(y)) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DropNormWavNet(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dims,
                 cond_audio_dropout=0.0,
                 cond_keyla_dropout=0.0,
                 use_cond_norm=False,
                 residual_layers=20,
                 residual_channels=512,
                 dilation_cycle_length=2,
                 pe_scale=1):
        super().__init__()
        self.cond_audio_dropout = cond_audio_dropout
        self.cond_keyla_dropout = cond_keyla_dropout
        self.use_cond_norm = use_cond_norm

        if use_cond_norm:
            self.cond_audio_norm = nn.GroupNorm(num_groups=residual_channels, num_channels=residual_channels)
            self.cond_keyla_norm = nn.GroupNorm(num_groups=residual_channels, num_channels=residual_channels)

        self.input_projection = Conv1d(in_dims, residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(residual_channels, pe_scale=pe_scale)
        dim = residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            NormResidualBlock(residual_channels, residual_channels, 2 ** (i % dilation_cycle_length), use_norm=use_cond_norm)
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, out_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, input, diffusion_step, cond_audio, cond_keyla, masks=None):
        """
        input: (B, d, T)
        diffusion_step: (B)
        cond: (B, d, T)
        """
        # cond1: audio
        # cond2: keyla
        if self.use_cond_norm:
            cond_audio = self.cond_audio_norm(cond_audio)
            cond_keyla = self.cond_keyla_norm(cond_keyla)

        if self.cond_audio_dropout > 0:
            cond_audio = F.dropout(cond_audio, self.cond_audio_dropout, self.training)
        if self.cond_keyla_dropout > 0:
            cond_keyla = F.dropout(cond_keyla, self.cond_keyla_dropout, self.training)

        cond = cond_audio + cond_keyla

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
        return x


def modulateseq(x, shift, scale):
    return x * (1 + scale) + shift


class AdaNormResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation, use_norm):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=2 * residual_channels, num_channels=2 * residual_channels) if use_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(num_groups=2 * residual_channels, num_channels=2 * residual_channels) if use_norm else nn.Identity()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.add_cond_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.mid_projection = Conv1d(2 * encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

        self.ada_modulation = nn.Sequential(
            nn.SiLU(),
            Conv1d(residual_channels, 4 * residual_channels, 1)
        )

    def forward(self, x, cond1, cond2, diffusion_step):
        # cond1: audio
        # cond2: keyla
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step

        shift_msa, scale_msa = self.ada_modulation(cond1).chunk(2, dim=1)

        y = self.norm1(self.dilated_conv(y)) + self.add_cond_projection(cond2)
        y = modulateseq(self.norm2(self.mid_projection(y)), shift_msa, scale_msa)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class AdaDropNormWavNet(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dims,
                 cond_audio_dropout=0.0,
                 cond_keyla_dropout=0.0,
                 use_cond_norm=False,
                 residual_layers=20,
                 residual_channels=512,
                 dilation_cycle_length=2,
                 pe_scale=1):
        super().__init__()
        self.cond_audio_dropout = cond_audio_dropout
        self.cond_keyla_dropout = cond_keyla_dropout
        self.use_cond_norm = use_cond_norm

        if use_cond_norm:
            self.cond_audio_norm = nn.GroupNorm(num_groups=residual_channels, num_channels=residual_channels)
            self.cond_keyla_norm = nn.GroupNorm(num_groups=residual_channels, num_channels=residual_channels)

        self.input_projection = Conv1d(in_dims, residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(residual_channels, pe_scale=pe_scale)
        dim = residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            AdaNormResidualBlock(residual_channels, residual_channels, 2 ** (i % dilation_cycle_length), use_norm=use_cond_norm)
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, out_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, input, diffusion_step, cond1, cond2, masks=None):
        """
        input: (B, d, T)
        diffusion_step: (B)
        cond: (B, d, T)
        """
        # cond1: audio
        # cond2: keyla
        if self.use_cond_norm:
            cond1 = self.cond_audio_norm(cond1)
            cond2 = self.cond_keyla_norm(cond2)

        if self.cond_audio_dropout > 0:
            cond1 = F.dropout(cond1, self.cond_audio_dropout, self.training)
        if self.cond_keyla_dropout > 0:
            cond2 = F.dropout(cond2, self.cond_keyla_dropout, self.training)

        x = self.input_projection(input)

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond1, cond2, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x


class AdaConcateNormResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation, use_norm):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=2 * residual_channels, num_channels=2 * residual_channels) if use_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(num_groups=2 * residual_channels, num_channels=2 * residual_channels) if use_norm else nn.Identity()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)

        self.concat_reduce = Conv1d(3 * residual_channels, 2 * residual_channels, 1)

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

        self.ada_modulation = nn.Sequential(
            nn.SiLU(),
            Conv1d(residual_channels, 4 * residual_channels, 1)
        )

    def forward(self, x, cond_audio, cond_keyla, diffusion_step):
        # cond1: audio
        # cond2: keyla
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step

        shift_msa, scale_msa = self.ada_modulation(cond_audio).chunk(2, dim=1)

        y = self.concat_reduce(torch.cat((self.norm1(self.dilated_conv(y)),
                                          cond_keyla), dim=1))
        y = modulateseq(self.norm2(y), shift_msa, scale_msa)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class AdaConcateDropNormWavNet(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dims,
                 cond_audio_dropout=0.0,
                 cond_keyla_dropout=0.0,
                 use_cond_norm=False,
                 residual_layers=20,
                 residual_channels=512,
                 dilation_cycle_length=2,
                 pe_scale=1):
        super().__init__()
        self.cond_audio_dropout = cond_audio_dropout
        self.cond_keyla_dropout = cond_keyla_dropout
        self.use_cond_norm = use_cond_norm

        if use_cond_norm:
            self.cond_audio_norm = nn.GroupNorm(num_groups=residual_channels, num_channels=residual_channels)
            self.cond_keyla_norm = nn.GroupNorm(num_groups=residual_channels, num_channels=residual_channels)

        self.input_projection = Conv1d(in_dims, residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(residual_channels, pe_scale=pe_scale)
        dim = residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            AdaConcateNormResidualBlock(residual_channels, residual_channels, 2 ** (i % dilation_cycle_length), use_norm=use_cond_norm)
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, out_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, input, diffusion_step, cond_audio, cond_keyla, masks=None):
        """
        input: (B, d, T)
        diffusion_step: (B)
        cond: (B, d, T)
        """
        # cond1: audio
        # cond2: keyla
        if self.use_cond_norm:
            cond_audio = self.cond_audio_norm(cond_audio)
            cond_keyla = self.cond_keyla_norm(cond_keyla)

        if self.cond_audio_dropout > 0:
            cond_audio = F.dropout(cond_audio, self.cond_audio_dropout, self.training)
        if self.cond_keyla_dropout > 0:
            cond_keyla = F.dropout(cond_keyla, self.cond_keyla_dropout, self.training)

        x = self.input_projection(input)

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond_audio, cond_keyla, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x