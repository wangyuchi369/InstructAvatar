#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : Tianyu He (xxxx@microsoft.com)
Date               : 2023-02-05 00:23
Last Modified By   : Tianyu He (xxxx@microsoft.com)
Last Modified Date : 2023-02-21 05:13
Description        : 
-------- 
Copyright (c) 2023 Microsoft Corporation.
'''

import os
from collections import OrderedDict
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_info

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from talking.modules.vqvae.mlp_encoder import TalkingMLPEncoderLayer


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear")
        if self.with_conv:
            x = self.conv(x)
        return x


class TalkingFuser(nn.Module):
    def __init__(self,
                 fusion_version,
                 embed_dim,
                 decoder_z_channels=None,
                 attn_num_heads=8,
                 attn_dropout=0.1):
        super().__init__()
        self.fusion_version = fusion_version
        # RGB feature and motion feature fusion
        if self.fusion_version == 'v1_cat-attn':
            # (batch, seq, feature)
            self.fuser = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                    dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                    batch_first=True)
            self.post_fuser_conv = nn.Conv1d(embed_dim+1, embed_dim, 1)
        elif self.fusion_version == 'v2_attn-upsample-attn':
            # (batch, seq, feature)
            self.fuser = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                    dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                    batch_first=True)
            self.upsample1_factor = 2
            # on last dim
            self.upsample1 = TalkingMLPEncoderLayer(embed_dim, embed_dim * self.upsample1_factor * 2, dropout=0)
            # (batch, seq, feature)
            self.fuser1 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            # channel first
            self.upsample2 = nn.Sequential(OrderedDict([
                                ('upsample2_1', Upsample(in_channels=embed_dim, with_conv=True)),  # 4x4
                                ('upsample2_2', Upsample(in_channels=embed_dim, with_conv=True)),  # 8x8
                                ('upsample2_3', Upsample(in_channels=embed_dim, with_conv=True)),  # 16x16
                             ]))
            # (batch, seq, feature)
            self.fuser2 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
        elif self.fusion_version == 'v3_16x16_attn' or self.fusion_version == 'v3_16x16_attn_wokeyframe':  # encode input to 16x16, direct attend to key frame feature
            # (batch, seq, feature)
            if 'wokeyframe' in fusion_version:
                rank_zero_info(f"[TalkingFuser] Skip attending to the first frame.")
            self.fuser = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                    dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                    batch_first=True)
            self.post_fuser_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        elif self.fusion_version == 'v4_4x4_attn-upsample-attn':
            # (batch, seq, feature)
            self.fuser1 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.upsample1 = Upsample(in_channels=embed_dim, with_conv=True)
            self.fuser2 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.upsample2 = Upsample(in_channels=embed_dim, with_conv=True)
            self.fuser3 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.post_fuser_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        elif self.fusion_version == 'v5_2x2_attn-upsample-attn':
            # (batch, seq, feature)
            self.fuser1 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.upsample1 = Upsample(in_channels=embed_dim, with_conv=True)
            self.fuser2 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.upsample2 = Upsample(in_channels=embed_dim, with_conv=True)
            self.fuser3 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.upsample3 = Upsample(in_channels=embed_dim, with_conv=True)
            self.fuser4 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.post_fuser_conv = nn.Conv2d(embed_dim, embed_dim, 1)
            z_shape = (1, embed_dim, 2, 2)
            rank_zero_info("[TalkingFuser] Working with z of shape {} = {} dimensions.".format(
                           z_shape, np.prod(z_shape)))
        elif self.fusion_version == 'v6_1x1_attn-upsample-attn':
            # for the feature of first frame
            encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=attn_num_heads,
                                                       dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                       batch_first=True)
            encoder_norm = nn.LayerNorm(embed_dim, eps=1e-5)
            self.key_frame_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, norm=encoder_norm)
            self.upsample0 = Upsample(in_channels=embed_dim, with_conv=True)
            # the following same as v5
            self.fuser1 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.upsample1 = Upsample(in_channels=embed_dim, with_conv=True)
            self.fuser2 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.upsample2 = Upsample(in_channels=embed_dim, with_conv=True)
            self.fuser3 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.upsample3 = Upsample(in_channels=embed_dim, with_conv=True)
            self.fuser4 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.post_fuser_conv = nn.Conv2d(embed_dim, embed_dim, 1)
            z_shape = (1, embed_dim, 1, 1)
            rank_zero_info("[TalkingFuser] Working with z of shape {} = {} dimensions.".format(
                           z_shape, np.prod(z_shape)))
        elif self.fusion_version == 'v7_1x1_no-crossattn':
            self.upsample1 = Upsample(in_channels=embed_dim, with_conv=True)
            self.upsample2 = Upsample(in_channels=embed_dim, with_conv=True)
            self.upsample3 = Decoder(ch=128, out_ch=embed_dim, ch_mult=(2,2,4), num_res_blocks=2,
                                     attn_resolutions=[16,8], dropout=0.0, in_channels=3,
                                     resolution=16, z_channels=embed_dim)
        elif self.fusion_version == 'v7-1_2x2_no-crossattn':
            # self.upsample1 = Upsample(in_channels=embed_dim, with_conv=True)
            self.upsample2 = Upsample(in_channels=embed_dim, with_conv=True)
            self.upsample3 = Decoder(ch=128, out_ch=embed_dim, ch_mult=(2,2,4), num_res_blocks=2,
                                     attn_resolutions=[16,8], dropout=0.0, in_channels=3,
                                     resolution=16, z_channels=embed_dim)
        elif self.fusion_version == 'v8_1x1_upsample-attn':
            # for the feature of first frame
            encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=attn_num_heads,
                                                       dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                       batch_first=True)
            encoder_norm = nn.LayerNorm(embed_dim, eps=1e-5)
            self.key_frame_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, norm=encoder_norm)
            # upsample
            self.upsample1 = Upsample(in_channels=embed_dim, with_conv=True)
            self.upsample2 = Upsample(in_channels=embed_dim, with_conv=True)
            self.upsample3 = Decoder(ch=128, out_ch=embed_dim, ch_mult=(2,2,4), num_res_blocks=2,
                                     attn_resolutions=[16,8], dropout=0.0, in_channels=3,
                                     resolution=16, z_channels=embed_dim)
            self.fuser4 = nn.TransformerDecoderLayer(embed_dim, nhead=attn_num_heads,
                                                     dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.post_fuser_conv = nn.Conv2d(embed_dim, embed_dim, 1)
            z_shape = (1, embed_dim, 1, 1)
            rank_zero_info("[TalkingFuser] Working with z of shape {} = {} dimensions.".format(
                           z_shape, np.prod(z_shape)))
        elif self.fusion_version == 'v9_2x2_upsample-attn':
            # for the feature of first frame
            encoder_layer = nn.TransformerEncoderLayer(decoder_z_channels, nhead=attn_num_heads,
                                                       dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                       batch_first=True)
            encoder_norm = nn.LayerNorm(decoder_z_channels, eps=1e-5)
            self.key_frame_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, norm=encoder_norm)
            # upsample
            self.conv1 = nn.Conv2d(embed_dim, decoder_z_channels, 1)
            self.upsample2 = Upsample(in_channels=decoder_z_channels, with_conv=True)
            self.upsample3 = Decoder(ch=128, out_ch=decoder_z_channels, ch_mult=(2,2,4), num_res_blocks=2,
                                     attn_resolutions=[16,8], dropout=0.0, in_channels=3,
                                     resolution=16, z_channels=decoder_z_channels)
            self.fuser4 = nn.TransformerDecoderLayer(decoder_z_channels, nhead=attn_num_heads,
                                                     dim_feedforward=decoder_z_channels*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.post_fuser_conv = nn.Conv2d(decoder_z_channels, decoder_z_channels, 1)
            z_shape = (1, embed_dim, 2, 2)
            rank_zero_info("[TalkingFuser] Working with z of shape {} = {} dimensions.".format(
                           z_shape, np.prod(z_shape)))
        elif self.fusion_version == 'v10_1x1_upsample-attn':
            # for the feature of first frame
            encoder_layer = nn.TransformerEncoderLayer(decoder_z_channels, nhead=attn_num_heads,
                                                       dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                       batch_first=True)
            encoder_norm = nn.LayerNorm(decoder_z_channels, eps=1e-5)
            self.key_frame_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, norm=encoder_norm)
            # upsample
            self.conv1 = nn.Conv2d(embed_dim, decoder_z_channels, 1)
            self.upsample1 = Upsample(in_channels=decoder_z_channels, with_conv=True)
            self.upsample2 = Upsample(in_channels=decoder_z_channels, with_conv=True)
            self.upsample3 = Decoder(ch=128, out_ch=decoder_z_channels, ch_mult=(2,2,4), num_res_blocks=2,
                                     attn_resolutions=[16,8], dropout=0.0, in_channels=3,
                                     resolution=16, z_channels=decoder_z_channels)
            self.fuser4 = nn.TransformerDecoderLayer(decoder_z_channels, nhead=attn_num_heads,
                                                     dim_feedforward=decoder_z_channels*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.post_fuser_conv = nn.Conv2d(decoder_z_channels, decoder_z_channels, 1)
            z_shape = (1, embed_dim, 1, 1)
            rank_zero_info("[TalkingFuser] Working with z of shape {} = {} dimensions.".format(
                           z_shape, np.prod(z_shape)))
        elif self.fusion_version == 'v11_4x4_upsample-attn':
            # for the feature of first frame
            encoder_layer = nn.TransformerEncoderLayer(decoder_z_channels, nhead=attn_num_heads,
                                                       dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                       batch_first=True)
            encoder_norm = nn.LayerNorm(decoder_z_channels, eps=1e-5)
            self.key_frame_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, norm=encoder_norm)
            # upsample
            self.conv1 = nn.Conv2d(embed_dim, decoder_z_channels, 1)
            self.upsample3 = Decoder(ch=128, out_ch=decoder_z_channels, ch_mult=(2,2,4), num_res_blocks=2,
                                     attn_resolutions=[16,8], dropout=0.0, in_channels=3,
                                     resolution=16, z_channels=decoder_z_channels)
            self.fuser4 = nn.TransformerDecoderLayer(decoder_z_channels, nhead=attn_num_heads,
                                                     dim_feedforward=decoder_z_channels*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.post_fuser_conv = nn.Conv2d(decoder_z_channels, decoder_z_channels, 1)
            z_shape = (1, embed_dim, 4, 4)
            rank_zero_info("[TalkingFuser] Working with z of shape {} = {} dimensions.".format(
                           z_shape, np.prod(z_shape)))
        elif self.fusion_version == 'v12_8x8_upsample-attn':
            # for the feature of first frame
            encoder_layer = nn.TransformerEncoderLayer(decoder_z_channels, nhead=attn_num_heads,
                                                       dim_feedforward=embed_dim*2, dropout=attn_dropout,
                                                       batch_first=True)
            encoder_norm = nn.LayerNorm(decoder_z_channels, eps=1e-5)
            self.key_frame_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, norm=encoder_norm)
            # upsample
            self.conv1 = nn.Conv2d(embed_dim, decoder_z_channels, 1)
            self.upsample3 = Decoder(ch=128, out_ch=decoder_z_channels, ch_mult=(2,4), num_res_blocks=2,
                                     attn_resolutions=[16,8], dropout=0.0, in_channels=3,
                                     resolution=16, z_channels=decoder_z_channels)
            self.fuser4 = nn.TransformerDecoderLayer(decoder_z_channels, nhead=attn_num_heads,
                                                     dim_feedforward=decoder_z_channels*2, dropout=attn_dropout,
                                                     batch_first=True)
            self.post_fuser_conv = nn.Conv2d(decoder_z_channels, decoder_z_channels, 1)
            z_shape = (1, embed_dim, 8, 8)
            rank_zero_info("[TalkingFuser] Working with z of shape {} = {} dimensions.".format(
                           z_shape, np.prod(z_shape)))
        else:
            raise NotImplementedError('[TalkingFuser] Not implemented fusion version')

    def fuse_before_decode_v1(self, quant, key_frame_quant_z):
        # fuse the input (e.g., exp coeff feature, motion feature, etc.) with the feature of first frame (RGB feature)
        # save the dim
        bs, key_frame_quant_z_h, key_frame_quant_z_w = key_frame_quant_z.shape[0], key_frame_quant_z.shape[2], key_frame_quant_z.shape[3]
        # rearrange key_frame_quant_z for serving as attention key & value
        key_frame_quant_z_to_memory = rearrange(key_frame_quant_z, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)
        quant = quant[:, :, 0, 0][:, None, :]  # (6, 1, 256)
        # concatenate the input (e.g., exp coeff feature, motion feature, etc.) and the feature of first frame (RGB feature)
        quant_and_key_frame_quant_z = torch.cat((quant, key_frame_quant_z_to_memory), dim=1)
        # conduct multi-head attention
        quant = self.fuser(tgt=quant_and_key_frame_quant_z, memory=key_frame_quant_z_to_memory)
        # reduce (6, 257, 256) to (6, 256, 256)
        quant = self.post_fuser_conv(quant)
        # get the final output for vqgan decoder input
        # TODO: switched permute and reshape, not run
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, key_frame_quant_z_h, key_frame_quant_z_w)  # (6, 256, 16, 16)
        return quant

    def fuse_before_decode_v2(self, quant, key_frame_quant_z):
        # fuse the input (e.g., exp coeff feature, motion feature, etc.) with the feature of first frame (RGB feature)
        # save the dim
        bs, key_frame_quant_z_h, key_frame_quant_z_w = key_frame_quant_z.shape[0], key_frame_quant_z.shape[2], key_frame_quant_z.shape[3]
        # rearrange key_frame_quant_z for serving as attention key & value
        key_frame_quant_z_to_memory = rearrange(key_frame_quant_z, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (6, 256, 256)
        quant = quant[:, :, 0, 0][:, None, :]  # (6, 1, 256)
        # conduct multi-head attention
        quant = self.fuser(tgt=quant, memory=key_frame_quant_z_to_memory)  # (6, 1, 256)
        assert quant.shape[1] == 1, "[TalkingFuser] Should be (B, 1, embed_dim)"
        # upsample1
        quant = self.upsample1(torch.squeeze(quant)).reshape(bs, self.upsample1_factor * 2, -1).contiguous()  # (6, 4, 256)
        # fuser1
        quant = self.fuser1(tgt=quant, memory=key_frame_quant_z_to_memory)  # (6, 4, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, self.upsample1_factor, self.upsample1_factor)  # (6, 256, 2, 2)
        # upsample2
        quant = self.upsample2(quant)  # (6, 256, 16, 16)
        # fuser2
        quant = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (6, 256, 256)
        quant = self.fuser2(tgt=quant, memory=key_frame_quant_z_to_memory)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, key_frame_quant_z_h, key_frame_quant_z_w)  # (6, 256, 16, 16)
        return quant

    def fuse_before_decode_v3(self, quant, key_frame_quant_z, fusion_version):
        # fuse the input (e.g., exp coeff feature, motion feature, etc.) with the feature of first frame (RGB feature)
        if 'wokeyframe' in fusion_version:
            # rank_zero_info(f"[TalkingFuser] Skip attending to the first frame.")
            quant = self.post_fuser_conv(quant)  # (6, 256, 16, 16)
            return quant
        # save the dim
        bs, key_frame_quant_z_h, key_frame_quant_z_w = key_frame_quant_z.shape[0], key_frame_quant_z.shape[2], key_frame_quant_z.shape[3]
        # rearrange key_frame_quant_z for serving as attention key & value
        key_frame_quant_z_to_memory = rearrange(key_frame_quant_z, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (6, 16x16, 256)
        # rearrange quant for serving as attention query
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (6, 16x16, 256)
        # conduct multi-head attention
        quant = self.fuser(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (6, 16x16, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, key_frame_quant_z_h, key_frame_quant_z_w)  # (6, 256, 16, 16)
        # after one conv
        quant = self.post_fuser_conv(quant)  # (6, 256, 16, 16)
        return quant

    def fuse_before_decode_v4(self, quant, key_frame_quant_z):
        # fuse the input (e.g., exp coeff feature, motion feature, etc.) with the feature of first frame (RGB feature)
        # save the dim
        bs, key_frame_quant_z_h, key_frame_quant_z_w = key_frame_quant_z.shape[0], key_frame_quant_z.shape[2], key_frame_quant_z.shape[3]
        # rearrange key_frame_quant_z for serving as attention key & value
        key_frame_quant_z_to_memory = rearrange(key_frame_quant_z, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        ############## 1
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 1
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 4x4, 256)
        # conduct multi-head attention 1
        quant = self.fuser1(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 4x4, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 4, 4)
        ############## 2
        # upsample 1
        quant = self.upsample1(quant)  # (bs, 256, 8, 8)
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 2
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 8x8, 256)
        # conduct multi-head attention 2
        quant = self.fuser2(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 8x8, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 8, 8)
        ############## 3
        # upsample 2
        quant = self.upsample2(quant)  # (bs, 256, 16, 16)
        # rearrange quant for serving as attention query 3
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        # conduct multi-head attention 3
        quant = self.fuser3(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 16x16, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, key_frame_quant_z_h, key_frame_quant_z_w)  # (bs, 256, 16, 16)
        ############## end
        # after one conv
        quant = self.post_fuser_conv(quant)  # (6, 256, 16, 16)
        return quant

    def fuse_before_decode_v5(self, quant, key_frame_quant_z):
        # fuse the input (e.g., exp coeff feature, motion feature, etc.) with the feature of first frame (RGB feature)
        # save the dim
        bs, key_frame_quant_z_h, key_frame_quant_z_w = key_frame_quant_z.shape[0], key_frame_quant_z.shape[2], key_frame_quant_z.shape[3]
        # rearrange key_frame_quant_z for serving as attention key & value
        key_frame_quant_z_to_memory = rearrange(key_frame_quant_z, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        ############## 1
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 1
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 2x2, 256)
        # conduct multi-head attention 1
        quant = self.fuser1(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 2x2, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 2, 2)
        ############## 2
        # upsample 1
        quant = self.upsample1(quant)  # (bs, 256, 4, 4)
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 2
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 4x4, 256)
        # conduct multi-head attention 2
        quant = self.fuser2(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 4x4, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 4, 4)
        ############## 3
        # upsample 2
        quant = self.upsample2(quant)  # (bs, 256, 8, 8)
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 3
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 8x8, 256)
        # conduct multi-head attention 3
        quant = self.fuser3(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 8x8, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 8, 8)
        ############## 4
        # upsample 3
        quant = self.upsample3(quant)  # (bs, 256, 16, 16)
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 4
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        # conduct multi-head attention 4
        quant = self.fuser4(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 16x16, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 16, 16)
        ############## end
        # confirm the dimension
        assert quant_h == key_frame_quant_z_h, quant_h
        # after one conv
        quant = self.post_fuser_conv(quant)  # (6, 256, 16, 16)
        return quant

    def fuse_before_decode_v6(self, quant, key_frame_quant_z):
        # fuse the input (e.g., exp coeff feature, motion feature, etc.) with the feature of first frame (RGB feature)
        # save the dim
        bs, key_frame_quant_z_h, key_frame_quant_z_w = key_frame_quant_z.shape[0], key_frame_quant_z.shape[2], key_frame_quant_z.shape[3]
        # rearrange key_frame_quant_z for serving as attention key & value
        key_frame_quant_z_to_memory = rearrange(key_frame_quant_z, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        # encode key_frame_quant
        key_frame_quant_z_to_memory = self.key_frame_encoder(key_frame_quant_z_to_memory)  # (bs, 16x16, 256)
        ############## 0 (bs, 256, 1, 1)
        # upsample 0
        quant = self.upsample0(quant)  # (bs, 256, 2, 2)
        ############## 1
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 1
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 2x2, 256)
        # conduct multi-head attention 1
        quant = self.fuser1(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 2x2, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 2, 2)
        ############## 2
        # upsample 1
        quant = self.upsample1(quant)  # (bs, 256, 4, 4)
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 2
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 4x4, 256)
        # conduct multi-head attention 2
        quant = self.fuser2(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 4x4, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 4, 4)
        ############## 3
        # upsample 2
        quant = self.upsample2(quant)  # (bs, 256, 8, 8)
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 3
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 8x8, 256)
        # conduct multi-head attention 3
        quant = self.fuser3(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 8x8, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 8, 8)
        ############## 4
        # upsample 3
        quant = self.upsample3(quant)  # (bs, 256, 16, 16)
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 4
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        # conduct multi-head attention 4
        quant = self.fuser4(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 16x16, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 16, 16)
        ############## end
        # confirm the dimension
        assert quant_h == key_frame_quant_z_h, quant_h
        # after one conv
        quant = self.post_fuser_conv(quant)  # (6, 256, 16, 16)
        return quant

    def fuse_before_decode_v7(self, quant):
        ############## 1 (bs, 256, 1, 1)
        # upsample 1
        quant = self.upsample1(quant)  # (bs, 256, 2, 2)
        ############## 2
        # upsample 2
        quant = self.upsample2(quant)  # (bs, 256, 4, 4)
        ############## 3
        # upsample 3
        quant = self.upsample3(quant)  # (bs, 256, 16, 16)
        return quant
    
    def fuse_before_decode_v7_1(self, quant):
        ############## 1 (bs, 256, 1, 1)
        # upsample 1
        # quant = self.upsample1(quant)  # (bs, 256, 2, 2)
        ############## 2
        # upsample 2
        quant = self.upsample2(quant)  # (bs, 256, 4, 4)
        ############## 3
        # upsample 3
        quant = self.upsample3(quant)  # (bs, 256, 16, 16)
        return quant

    def fuse_before_decode_v8(self, quant, key_frame_quant_z):
        # fuse the input (e.g., exp coeff feature, motion feature, etc.) with the feature of first frame (RGB feature)
        # save the dim
        bs, key_frame_quant_z_h, key_frame_quant_z_w = key_frame_quant_z.shape[0], key_frame_quant_z.shape[2], key_frame_quant_z.shape[3]
        # rearrange key_frame_quant_z for serving as attention key & value
        key_frame_quant_z_to_memory = rearrange(key_frame_quant_z, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        # encode key_frame_quant
        key_frame_quant_z_to_memory = self.key_frame_encoder(key_frame_quant_z_to_memory)  # (bs, 16x16, 256)
        ############## 1 (bs, 256, 1, 1)
        # upsample 1
        quant = self.upsample1(quant)  # (bs, 256, 2, 2)
        ############## 2
        # upsample 2
        quant = self.upsample2(quant)  # (bs, 256, 4, 4)
        ############## 3
        # upsample 3
        quant = self.upsample3(quant)  # (bs, 256, 16, 16)
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 4
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        # conduct multi-head attention 4
        quant = self.fuser4(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 16x16, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 16, 16)
        ############## end
        # confirm the dimension
        assert quant_h == key_frame_quant_z_h, quant_h
        # after one conv
        quant = self.post_fuser_conv(quant)  # (6, 256, 16, 16)
        return quant

    def fuse_before_decode_v9(self, quant, key_frame_quant_z):
        # fuse the input (e.g., exp coeff feature, motion feature, etc.) with the feature of first frame (RGB feature)
        # save the dim
        bs, key_frame_quant_z_h, key_frame_quant_z_w = key_frame_quant_z.shape[0], key_frame_quant_z.shape[2], key_frame_quant_z.shape[3]
        # rearrange key_frame_quant_z for serving as attention key & value
        key_frame_quant_z_to_memory = rearrange(key_frame_quant_z, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        # encode key_frame_quant
        key_frame_quant_z_to_memory = self.key_frame_encoder(key_frame_quant_z_to_memory)  # (bs, 16x16, 256)
        ############## 1 (bs, 256, 1, 1)
        # upsample 1
        quant = self.conv1(quant)  # (bs, 256, 2, 2)
        ############## 2
        # upsample 2
        quant = self.upsample2(quant)  # (bs, 256, 4, 4)
        ############## 3
        # upsample 3
        quant = self.upsample3(quant)  # (bs, 256, 16, 16)
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 4
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        # conduct multi-head attention 4
        quant = self.fuser4(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 16x16, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 16, 16)
        ############## end
        # confirm the dimension
        assert quant_h == key_frame_quant_z_h, quant_h
        # after one conv
        quant = self.post_fuser_conv(quant)  # (6, 256, 16, 16)
        return quant

    def fuse_before_decode_v10(self, quant, key_frame_quant_z):
        # fuse the input (e.g., exp coeff feature, motion feature, etc.) with the feature of first frame (RGB feature)
        # save the dim
        bs, key_frame_quant_z_h, key_frame_quant_z_w = key_frame_quant_z.shape[0], key_frame_quant_z.shape[2], key_frame_quant_z.shape[3]
        # rearrange key_frame_quant_z for serving as attention key & value
        key_frame_quant_z_to_memory = rearrange(key_frame_quant_z, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        # encode key_frame_quant
        key_frame_quant_z_to_memory = self.key_frame_encoder(key_frame_quant_z_to_memory)  # (bs, 16x16, 256)
        ############## 1 (bs, 256, 1, 1)
        # upsample 1
        quant = self.conv1(quant)
        quant = self.upsample1(quant)  # (bs, 256, 4, 4)
        ############## 2
        # upsample 2
        quant = self.upsample2(quant)  # (bs, 256, 4, 4)
        ############## 3
        # upsample 3
        quant = self.upsample3(quant)  # (bs, 256, 16, 16)
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 4
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        # conduct multi-head attention 4
        quant = self.fuser4(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 16x16, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 16, 16)
        ############## end
        # confirm the dimension
        assert quant_h == key_frame_quant_z_h, quant_h
        # after one conv
        quant = self.post_fuser_conv(quant)  # (6, 256, 16, 16)
        return quant

    def fuse_before_decode_v11_12(self, quant, key_frame_quant_z):
        # fuse the input (e.g., exp coeff feature, motion feature, etc.) with the feature of first frame (RGB feature)
        # save the dim
        bs, key_frame_quant_z_h, key_frame_quant_z_w = key_frame_quant_z.shape[0], key_frame_quant_z.shape[2], key_frame_quant_z.shape[3]
        # rearrange key_frame_quant_z for serving as attention key & value
        key_frame_quant_z_to_memory = rearrange(key_frame_quant_z, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        # encode key_frame_quant
        key_frame_quant_z_to_memory = self.key_frame_encoder(key_frame_quant_z_to_memory)  # (bs, 16x16, 256)
        ############## 1 (bs, 256, 1, 1)
        # upsample 1
        quant = self.conv1(quant)
        ############## 2
        # upsample 2
        ############## 3
        # upsample 3
        # (bs, 256, 4, 4)
        quant = self.upsample3(quant)  # (bs, 256, 16, 16)
        # save the dim
        quant_h, quant_w = quant.shape[2], quant.shape[3]
        # rearrange quant for serving as attention query 4
        quant_seq = rearrange(quant, 'b c h w -> b (h w) c').to(memory_format=torch.contiguous_format)  # (bs, 16x16, 256)
        # conduct multi-head attention 4
        quant = self.fuser4(tgt=quant_seq, memory=key_frame_quant_z_to_memory)  # (bs, 16x16, 256)
        quant = quant.permute(0, 2, 1).contiguous().reshape(bs, -1, quant_h, quant_w)  # (bs, 256, 16, 16)
        ############## end
        # confirm the dimension
        assert quant_h == key_frame_quant_z_h, quant_h
        # after one conv
        quant = self.post_fuser_conv(quant)  # (6, 256, 16, 16)
        return quant

    def forward(self, quant, key_frame_quant_z):
        if self.fusion_version == 'v1_cat-attn':
            quant = self.fuse_before_decode_v1(quant, key_frame_quant_z)
        elif self.fusion_version == 'v2_attn-upsample-attn':
            quant = self.fuse_before_decode_v2(quant, key_frame_quant_z)
        elif self.fusion_version == 'v3_16x16_attn' or self.fusion_version == 'v3_16x16_attn_wokeyframe':
            quant = self.fuse_before_decode_v3(quant, key_frame_quant_z, self.fusion_version)
        elif self.fusion_version == 'v4_4x4_attn-upsample-attn':
            quant = self.fuse_before_decode_v4(quant, key_frame_quant_z)
        elif self.fusion_version == 'v5_2x2_attn-upsample-attn':
            quant = self.fuse_before_decode_v5(quant, key_frame_quant_z)
        elif self.fusion_version == 'v6_1x1_attn-upsample-attn':
            quant = self.fuse_before_decode_v6(quant, key_frame_quant_z)
        elif self.fusion_version == 'v7_1x1_no-crossattn':
            quant = self.fuse_before_decode_v7(quant)
        elif self.fusion_version == 'v7-1_2x2_no-crossattn':
            quant = self.fuse_before_decode_v7_1(quant)
        elif self.fusion_version == 'v8_1x1_upsample-attn':
            quant = self.fuse_before_decode_v8(quant, key_frame_quant_z)
        elif self.fusion_version == 'v9_2x2_upsample-attn':
            quant = self.fuse_before_decode_v9(quant, key_frame_quant_z)
        elif self.fusion_version == 'v10_1x1_upsample-attn':
            quant = self.fuse_before_decode_v10(quant, key_frame_quant_z)
        elif self.fusion_version == 'v11_4x4_upsample-attn' or self.fusion_version == 'v12_8x8_upsample-attn':
            quant = self.fuse_before_decode_v11_12(quant, key_frame_quant_z)
        else:
            raise NotImplementedError('[TalkingFuser] Not implemented fusion version')
        return quant