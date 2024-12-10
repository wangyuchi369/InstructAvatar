import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.pytorch.utilities import rank_zero_info
# import torch.utils.checkpoint as checkpoint
from ldm.modules.diffusionmodules.util import checkpoint
from ldm.modules.diffusionmodules.model import make_attn, Upsample, Downsample, Normalize, nonlinearity
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.diffusionmodules.model import ResnetBlock as LegacyResnetBlock
from talking.modules.vqvae_2enc.temporal_modules import make_attn_temporal
from talking.modules.vqvae_2enc.util import Hourglass, make_coordinate_grid, LayerNorm2d, get_nonspade_norm_layer
from einops import rearrange


class ResnetBlock(nn.Module):
    def __init__(self, *, gradient_checkpointing, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
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

    # https://github.com/lucidrains/DALLE2-pytorch/issues/225
    # def forward(self, x):
    #     if self.gradient_checkpointing and self.training:
    #         return checkpoint.checkpoint(self._forward, x)
    #     else:
    #         return self._forward(x)

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), self.gradient_checkpointing)

    def _forward(self, x):
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


class SpatioTemporalResnetBlock(ResnetBlock):
    
    def __init__(self, gradient_checkpointing=False, temporal_kernel_size = 3, temporal_len=5, **kwargs):

        super().__init__(gradient_checkpointing=gradient_checkpointing, **kwargs)
        self.temporal_kernel_size = temporal_kernel_size
        self.gradient_checkpointing = gradient_checkpointing
        self.padding = (self.temporal_kernel_size - 1) // 2
        self.temporal_len = temporal_len
        self.temporal_conv_1 = torch.nn.Conv1d(self.out_channels, self.out_channels, kernel_size=self.temporal_kernel_size, stride=1, padding=self.padding)
        self.temporal_conv_2 = torch.nn.Conv1d(self.out_channels, self.out_channels,
                                               kernel_size=self.temporal_kernel_size, stride=1, padding=self.padding)
    def forward(self, x):
        to_be_updated = list(self.temporal_conv_1.parameters()) + list(self.temporal_conv_2.parameters())
        return checkpoint(
            self._forward, (x,), to_be_updated, self.gradient_checkpointing)
        #return self._forward(x)
    def _forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # if temb is not None:
        #     h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        # add temporal conv
        resolution = h.shape[-1]
        temp_input = rearrange(h, '(b t) c h w -> (b h w) c t', t=self.temporal_len)
        temp_input = self.temporal_conv_1(temp_input)
        temp_output = rearrange(temp_input, '(b h w) c t -> (b t) c h w', h=resolution, w=resolution)
        h = h + temp_output

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        temp_input = rearrange(h, '(b t) c h w -> (b h w) c t', t=self.temporal_len)
        temp_input = self.temporal_conv_2(temp_input)
        temp_output = rearrange(temp_input, '(b h w) c t -> (b t) c h w', h=resolution, w=resolution)
        h = h + temp_output

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class LegacyEncoder(Encoder):
    def forward(self, x, return_hs=False):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if return_hs:
            return h, hs
        else:
            return h

class LegacyDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, use_unet_connection=False,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.use_unet_connection = use_unet_connection

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = LegacyResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = LegacyResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        self.unet_connect_proj = nn.ModuleList()
        input_ch_list = [128] * 4 + [256] * 3 + [512] * 2
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if use_unet_connection:
                    self.unet_connect_proj.insert(0, nn.Conv2d(block_in + input_ch_list.pop(),
                                                    block_in,
                                                    1))
                block.append(LegacyResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            
            if i_level != 0:
                # if use_unet_connection:
                #     unet_proj.upsample = nn.Conv2d(2*block_in,
                #                                      block_in,
                #                                      1)
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            
            self.up.insert(0, up) # prepend to get consistent order
            # self.unet_connect_proj.insert(0, unet_proj)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, hs=None):
        if hs is not None:
            assert self.use_unet_connection, "hs is not None but unet connection is not used"
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        
        # upsampling
        proj_index = len(self.unet_connect_proj) - 1
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                if hs is not None:
                    h = torch.cat([h, hs.pop()], dim=1)
                    h = self.unet_connect_proj[proj_index](h)
                    proj_index -= 1
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                # if hs is not None:
                #     print(i_level, h.size(), hs[-1].size())
                #     print("===in upsample===")
                #     h = torch.cat([h, hs.pop()], dim=1)
                #     h = self.unet_connect_proj[i_level].upsample(h)
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class Encoder(nn.Module):
    def __init__(self, *, gradient_checkpointing, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(gradient_checkpointing=gradient_checkpointing,
                                             in_channels=block_in,
                                             out_channels=block_out,
                                             dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(gradient_checkpointing=gradient_checkpointing,
                                           in_channels=block_in,
                                           out_channels=block_in,
                                           dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(gradient_checkpointing=gradient_checkpointing,
                                           in_channels=block_in,
                                           out_channels=block_in,
                                           dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, return_hs=False):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if return_hs:
            return h, hs
        else:
            return h


class EncoderWithTemporal(nn.Module):
    def __init__(self, *, gradient_checkpointing, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 temporal_kernel_size=None,  temporal_len=None,
                 **ignore_kwargs):
        super().__init__()
        assert temporal_kernel_size is not None
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_len = temporal_len
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
        
                block.append(SpatioTemporalResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             dropout=dropout,
                                            temporal_kernel_size=self.temporal_kernel_size,
                                            temporal_len=self.temporal_len,
                                            gradient_checkpointing=gradient_checkpointing))  
              
                block_in = block_out
                if curr_res in attn_resolutions:
                    # TODO may add temporal attn in other resolutions too
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        
        self.mid.block_1 = SpatioTemporalResnetBlock(in_channels=block_in,
                                    out_channels=block_in,
                                    temporal_kernel_size=self.temporal_kernel_size,
                                    dropout=dropout,
                                    temporal_len=self.temporal_len,
                                    gradient_checkpointing=gradient_checkpointing)
        self.mid.attn_1 = make_attn_temporal(block_in, attn_type=attn_type, temporal_len=self.temporal_len, use_temporal=True)
        self.mid.block_2 = SpatioTemporalResnetBlock(in_channels=block_in,
                                    out_channels=block_in,
                                    temporal_kernel_size=self.temporal_kernel_size,
                                    dropout=dropout,
                                    temporal_len=self.temporal_len,
                                    gradient_checkpointing=gradient_checkpointing)
    

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, return_hs=False):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if return_hs:
            return h, hs
        else:
            return h


class Decoder(nn.Module):
    def __init__(self, *, gradient_checkpointing, ch, out_ch, num_res_blocks,
                 attn_resolutions, in_channels, resolution, z_channels,
                 ch_mult=(1,2,4,8), dropout=0.0, resamp_with_conv=True,
                 give_pre_end=False, tanh_out=False, use_linear_attn=False, use_unet_connection=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(gradient_checkpointing=gradient_checkpointing,
                                           in_channels=block_in,
                                           out_channels=block_in,
                                           dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(gradient_checkpointing=gradient_checkpointing,
                                           in_channels=block_in,
                                           out_channels=block_in,
                                           dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(gradient_checkpointing=gradient_checkpointing,
                                             in_channels=block_in,
                                             out_channels=block_out,
                                             dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, hs=None):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class DecoderWithTemporal(nn.Module):
    def __init__(self, *, gradient_checkpointing, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 temporal_kernel_size=None, temporal_len=None,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        assert temporal_kernel_size is not None
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_len = temporal_len
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
       
        self.mid.block_1 = SpatioTemporalResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        dropout=dropout,
                                        temporal_len=self.temporal_len,
                                        gradient_checkpointing=gradient_checkpointing)
        self.mid.attn_1 = make_attn_temporal(block_in, attn_type=attn_type, temporal_len=self.temporal_len, use_temporal=True)
        self.mid.block_2 = SpatioTemporalResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        dropout=dropout,
                                        temporal_len=self.temporal_len,
                                    gradient_checkpointing=gradient_checkpointing)
        # else:
        #     self.mid.block_1 = ResnetBlock(gradient_checkpointing=gradient_checkpointing,
        #                                     in_channels=block_in,
        #                                     out_channels=block_in,
        #                                     dropout=dropout)
        #     self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        #     self.mid.block_2 = ResnetBlock(gradient_checkpointing=gradient_checkpointing,
        #                                    in_channels=block_in,
        #                                    out_channels=block_in,
        #                                    dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
              
                block.append(SpatioTemporalResnetBlock(in_channels=block_in,
                                                out_channels=block_out,
                                                dropout=dropout,
                                                temporal_len=self.temporal_len,
                                                gradient_checkpointing=gradient_checkpointing))
           
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, hs=None):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class WarppedDecoder(nn.Module):
    def __init__(self, *, gradient_checkpointing, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        rank_zero_info("[WarppedDecoder] Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(gradient_checkpointing=gradient_checkpointing,
                                           in_channels=block_in,
                                           out_channels=block_in,
                                           dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(gradient_checkpointing=gradient_checkpointing,
                                           in_channels=block_in,
                                           out_channels=block_in,
                                           dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(gradient_checkpointing=gradient_checkpointing,
                                             in_channels=block_in,
                                             out_channels=block_out,
                                             dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # warpping modules
        norm_layer = functools.partial(LayerNorm2d, affine=True)
        self.pre_warp = nn.Sequential(
            norm_layer(ch*ch_mult[0]),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(ch*ch_mult[0], 2, kernel_size=7, stride=1, padding=3),
        )

        # post warp
        post_warp_ch = 32
        self.post_warp_conv_in = torch.nn.Conv2d(3,
                                                 post_warp_ch,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1)
        self.post_warp = nn.Module()
        # self.post_warp.block_1 = ResnetBlock(gradient_checkpointing=gradient_checkpointing,
        #                                          in_channels=post_warp_ch,
        #                                          out_channels=post_warp_ch,
        #                                          dropout=dropout)
        self.post_warp.attn_1 = make_attn(post_warp_ch, attn_type=attn_type)
        self.post_warp.block_2 = ResnetBlock(gradient_checkpointing=gradient_checkpointing,
                                                 in_channels=post_warp_ch,
                                                 out_channels=post_warp_ch,
                                                 dropout=dropout)

        # end
        self.norm_out = Normalize(post_warp_ch)
        self.conv_out = torch.nn.Conv2d(post_warp_ch,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def _deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode="bilinear")
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(
            inp.to(deformation.dtype), deformation, padding_mode="reflection"
        )

    def forward(self, z, keyframe_img):
        # z: (bs, 3, 64, 64)
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # h: (bs, 128, 256, 256)
        # pre warp
        h = self.pre_warp(h)  # h: (bs, 2, 256, 256)

        _, _, h_, w_ = h.shape
        h = 2 * torch.cat(
            [h[:, :1, ...] / (w_ - 1), h[:, 1:, ...] / (h_ - 1)], 1
        )
        grid = make_coordinate_grid((h_, w_), type=torch.FloatTensor).to(h.device)
        h = grid + h.permute(0, 2, 3, 1)

        # keyframe_img: (bs, 3, 256, 256)
        deformed = self._deform_input(keyframe_img, h)  # (bs, 3, 256, 256)

        # post warp
        h = self.post_warp_conv_in(deformed)
        # h = self.post_warp.block_1(h)
        h = self.post_warp.attn_1(h)
        h = self.post_warp.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        h = h + deformed
        return h, deformed


class LadderEncoder(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, need_feat=False, in_dims=3, z_dim=512, norm_type="spectralinstance"):
        super().__init__()
        self.need_feat = need_feat

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        nef = 64
        norm_layer = get_nonspade_norm_layer(norm_type)
        self.layer1 = norm_layer(nn.Conv2d(in_dims, nef, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(nef * 1, nef * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(nef * 2, nef * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(nef * 4, nef * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(nef * 8, nef * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(nef * 8, nef * 8, kw, stride=2, padding=pw))

        if need_feat:
            self.up_layer2 = norm_layer(
                nn.Conv2d(nef * 2, nef * 2, kw, stride=1, padding=pw)
            )
            self.up_layer3 = nn.Sequential(
                norm_layer(nn.Conv2d(nef * 4, nef * 2, kw, stride=1, padding=pw)),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            )
            self.up_layer4 = nn.Sequential(
                norm_layer(nn.Conv2d(nef * 8, nef * 2, kw, stride=1, padding=pw)),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            )
            self.up_layer5 = nn.Sequential(
                norm_layer(nn.Conv2d(nef * 8, nef * 2, kw, stride=1, padding=pw)),
                nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            )
            self.up_layer6 = nn.Sequential(
                norm_layer(nn.Conv2d(nef * 8, nef * 2, kw, stride=1, padding=pw)),
                nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
            )

        self.actvn = nn.LeakyReLU(0.2, False)
        self.so = s0 = 4
        self.fc = nn.Linear(nef * 8 * s0 * s0, z_dim)

    def forward(self, x):
        features = None
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode="bilinear")

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        if self.need_feat:
            features = self.up_layer2(x)
        x = self.layer3(self.actvn(x))
        if self.need_feat:
            features = self.up_layer3(x) + features
        x = self.layer4(self.actvn(x))
        if self.need_feat:
            features = self.up_layer4(x) + features
        x = self.layer5(self.actvn(x))
        if self.need_feat:
            features = self.up_layer5(x) + features
        x = self.layer6(self.actvn(x))
        if self.need_feat:
            features = self.up_layer6(x) + features

        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x / (x.norm(dim=-1, p=2, keepdim=True) + 1e-5)

        return x, features


class DenseMotionNetworkReg(nn.Module):
    def __init__(
        self,
        block_expansion,
        num_blocks,
        max_features,
        Lwarp=False,
        AdaINc=0,
        dec_lease=0,
        in_dim=0,
    ):
        super(DenseMotionNetworkReg, self).__init__()
        self.hourglass = Hourglass(
            block_expansion=block_expansion,
            in_features=in_dim,
            max_features=max_features,
            num_blocks=num_blocks,
            Lwarp=Lwarp,
            AdaINc=AdaINc,
            dec_lease=dec_lease,
        )

        if dec_lease > 0:
            norm_layer = functools.partial(LayerNorm2d, affine=True)
            self.reger = nn.Sequential(
                norm_layer(self.hourglass.out_filters),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    self.hourglass.out_filters, 2, kernel_size=7, stride=1, padding=3
                ),
            )
        else:
            self.reger = nn.Conv2d(
                self.hourglass.out_filters, 2, kernel_size=(7, 7), padding=(3, 3)
            )

    def forward(self, source_image, drv_deca):
        # source_image: (bs, 9, 256, 256)
        # drv_deca: (bs, 512)
        prediction = self.hourglass(source_image, drv_exp=drv_deca)
        # prediction: (bs, 512, 64, 64)
        out_dict = {}
        flow = self.reger(prediction)
        # flow: (bs, 2, 64, 64)
        bs, _, h, w = flow.shape
        flow_norm = 2 * torch.cat(
            [flow[:, :1, ...] / (w - 1), flow[:, 1:, ...] / (h - 1)], 1
        )
        # flow_norm: (bs, 2, 64, 64)
        out_dict["flow"] = flow_norm
        grid = make_coordinate_grid((h, w), type=torch.FloatTensor).to(flow_norm.device)
        deformation = grid + flow_norm.permute(0, 2, 3, 1)
        # grid: (64, 64, 2)
        # deformation: (bs, 64, 64, 2)
        out_dict["deformation"] = deformation
        return out_dict