#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : Junliang Guo
Date               : 2023-05-10 19:44
Last Modified By   : Tianyu He (xxxx@microsoft.com)
Last Modified Date : 2023-05-22 05:03
Description        : pytorch lightning < 2.0.0
-------- 
Copyright (c) 2023 Microsoft Corporation.
'''

import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import numpy as np
from contextlib import contextmanager
from packaging import version

from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.ema import LitEma

from transformers import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoder

from ldm.modules.diffusionmodules.model import ResnetBlock, Downsample, Upsample
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config

from talking.modules.vqvae_2enc.encdec import Encoder as EncoderCKPT
from talking.modules.vqvae_2enc.encdec import LegacyEncoder as Encoder
from talking.modules.vqvae_2enc.encdec import Decoder as DecoderCKPT
from talking.modules.vqvae_2enc.encdec import LegacyDecoder as Decoder
from talking.modules.vqvae_2enc.residual_vq import ResidualVQ

from talking.modules.vqvae_2enc.encdec import EncoderWithTemporal as EncoderWithTemporalCKPT
from talking.modules.vqvae_2enc.encdec import DecoderWithTemporal as DecoderWithTemporalCKPT
# from talking.modules.vqvae_2enc.temporal_modules import Encoder as TemporalEncoder
# from talking.modules.vqvae_2enc.temporal_modules import Decoder as TemporalDecoder
from talking.modules.vqvae_2enc.smooth import moving_average, one_euro_smooth

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

class VQModelwithLDMK(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,   # 8192  
                 embed_dim,     # 3
                 vq_num_quantizers=1, # add
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 num_down=2,    # add
                 logit_scale=20.0,
                 first_init=False,  
                 cross_concat=False,    # forward appear和ldmk是否cross
                 *args, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.vq_num_quantizers = vq_num_quantizers
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.ldmk_encoder = Encoder(**ddconfig) # add
        self.loss = instantiate_from_config(lossconfig)
        # ResidualVQ
        self.quantizer_appear = ResidualVQ(num_quantizers=vq_num_quantizers,
                                    n_e=n_embed, e_dim=embed_dim, beta=0.25,
                                    remap=remap,
                                    sane_index_shape=sane_index_shape,
                                    legacy=False)
        self.quantizer_ldmk = ResidualVQ(num_quantizers=vq_num_quantizers,
                                    n_e=n_embed, e_dim=embed_dim, beta=0.25,
                                    remap=remap,
                                    sane_index_shape=sane_index_shape,
                                    legacy=False)
        # self.quantizer_appear = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        #                                 remap=remap,
        #                                 sane_index_shape=sane_index_shape,
        #                                 legacy=False)       # TODO: add
        # self.quantizer_ldmk = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        #                                 remap=remap,
        #                                 sane_index_shape=sane_index_shape,
        #                                 legacy=False)       # TODO: add
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.quant_conv_ldmk = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.quant_ldmk_and_conv = torch.nn.Conv2d(2*embed_dim, embed_dim, 1)
        self.cross_concat = cross_concat
        self.logit_scale = logit_scale


        downs = []
        for _ in range(num_down):
            downs.append(Downsample(ddconfig["z_channels"], True))
            # downs.append(Downsample(2*ddconfig["z_channels"], True))
            # appear_hw = appear_hw // 2
        
        ups = []
        for _ in range(num_down):       # /2 /2
            ups.append(Upsample(ddconfig["z_channels"], True))
            # appear_hw = appear_hw * 2

        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)
        
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        if ckpt_path is not None:
            self.first_init = first_init
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor


    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)  # quantize
        return h
    
    def encode_ldmk(self, x):
        ldmk_h = self.ldmk_encoder(x) 

        for down in self.downs:
            ldmk_h = down(ldmk_h)

        moments = self.quant_conv_ldmk(ldmk_h) 
        ldmk_h = moments    # not sample

        ldmk_h, ldmk_h_commit_loss, _ = self.quantizer_ldmk(ldmk_h)

        for up in self.ups:
            ldmk_h = up(ldmk_h)

        return ldmk_h, ldmk_h_commit_loss

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, 
                ldmks=None,
                concat_input=False):
        
        z_appear = self.encode(input) 
        z_appear, appear_commit_loss, _ = self.quantizer_appear(z_appear)
        z_ldmk, ldmk_commit_loss = self.encode_ldmk(ldmks)

        commit_loss = appear_commit_loss + ldmk_commit_loss
        # print("1 2 sum", appear_commit_loss, ldmk_commit_loss, commit_loss)

        if concat_input:
            z_appear_input, z_appear_pos = torch.chunk(z_appear, 2, dim=0)
            z_ldmk_inputs, z_ldmk_pos = torch.chunk(z_ldmk, 2, dim=0)
            if self.cross_concat:
                z = torch.cat([z_appear_pos, z_ldmk_inputs], dim=1)
            else:
                z = torch.cat([z_appear_input, z_ldmk_inputs], dim=1)
        else:
            z = torch.cat([z_appear, z_ldmk], dim=1)
        
        # concat: bsz x 6 x 64 x 64, quant: bsz x 3 x 64 x 64
        z = self.quant_ldmk_and_conv(z) 
        
        dec = self.decode(z)

        return dec, commit_loss

    def _get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_input(self, batch, k):
        x = self._get_input(batch, "image")
        x_pos = self._get_input(batch, "positive_img")
        x_ldmk = self._get_input(batch, "ldmk_img")
        x_ldmk_pos = self._get_input(batch, "ldmk_pos_img")
        return x, x_pos, x_ldmk, x_ldmk_pos

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs, inputs_pos, ldmks, ldmks_pos = self.get_input(batch, self.image_key)
        inputs_pos = inputs_pos.to(self.device)
        ldmks = ldmks.to(self.device)
        # print("training input_pos", inputs_pos.shape, inputs_pos.dtype)
        concat_inputs = torch.cat([inputs, inputs_pos], dim=0)
        concat_ldmks = torch.cat([ldmks, ldmks_pos], dim=0)

        reconstructions, commit_loss = self(inputs_pos, ldmks=ldmks)

        if optimizer_idx == 0:
            # train encoder+decoder+codebook
            aeloss, log_dict_ae = self.loss(commit_loss, inputs, reconstructions, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            # print(aeloss, log_dict_ae)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(commit_loss, inputs, reconstructions, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            # print(discloss, log_dict_disc)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs, inputs_pos, ldmks, ldmks_pos = self.get_input(batch, self.image_key)
        inputs_pos = inputs_pos.to(self.device)
        ldmks = ldmks.to(self.device)
        # print("validation input_pos", inputs_pos.shape, inputs_pos.dtype)
        concat_inputs = torch.cat([inputs, inputs_pos], dim=0)
        concat_ldmks = torch.cat([ldmks, ldmks_pos], dim=0)

        reconstructions, commit_loss = self(inputs_pos, ldmks=ldmks)

        aeloss, log_dict_ae = self.loss(commit_loss, inputs, reconstructions, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(commit_loss, inputs, reconstructions, 1, self.global_step,
                                    last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.ldmk_encoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.quant_ldmk_and_conv.parameters())+
                                  list(self.quant_conv_ldmk.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.ups.parameters())+
                                  list(self.downs.parameters())+
                                  list(self.quantizer_ldmk.parameters())+
                                  list(self.quantizer_appear.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        inputs, inputs_pos, ldmks, ldmks_pos = self.get_input(batch, self.image_key)
        inputs_pos = inputs_pos.to(self.device)
        ldmks = ldmks.to(self.device)
        # inputs_pos = torch.as_tensor(inputs_pos, dtype=torch.float16).to(self.device)
        # ldmks = torch.as_tensor(ldmks, dtype=torch.float16).to(self.device)
        # print("input_pos", inputs_pos.shape, inputs_pos.dtype)
        concat_inputs = torch.cat([inputs, inputs_pos], dim=0).to(self.device)
        concat_ldmks = torch.cat([ldmks, ldmks_pos], dim=0).to(self.device)

        x = inputs.to(self.device)
        if not only_inputs:
            xrec, _ = self(inputs_pos, ldmks=ldmks)
            # xrec, _, _ = self(concat_inputs, ldmks=concat_ldmks)
            # xrec, posterior, _, _ = self(concat_inputs, ldmks=concat_ldmks)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            # log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]

        if self.first_init:
            print("First init, copy the encoder parameters to ldmk_encoder.")
            old_keys = list(sd.keys())
            for k in old_keys:
                if k.startswith("encoder"):
                    new_k = k.replace("encoder", "ldmk_encoder")
                    sd[new_k] = sd[k]

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 gradient_checkpointing=False,
                 ):
        super().__init__()
        self.image_key = image_key
        if gradient_checkpointing:
            self.encoder = EncoderCKPT(gradient_checkpointing=gradient_checkpointing, **ddconfig)
            self.decoder = DecoderCKPT(gradient_checkpointing=gradient_checkpointing, **ddconfig)
        else:
            self.encoder = Encoder(**ddconfig)
            self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    
class AutoEncoderKLwithLDMK(AutoencoderKL):
    def __init__(self, ddconfig, 
                 num_down=2,
                 logit_scale=20.0,
                 first_init=False,
                 cross_concat=False,
                 gather_with_grad=False,
                 ldmk_kl_loss=False,
                 use_head_pose=False,
                 use_ae=False,
                 gradient_checkpointing=False,
                 use_unet_connection=False,
                 *args, **kwargs):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(ddconfig, gradient_checkpointing=gradient_checkpointing, *args, **kwargs)
        if gradient_checkpointing:
            self.ldmk_encoder = EncoderCKPT(gradient_checkpointing=gradient_checkpointing, **ddconfig)
        else:
            self.ldmk_encoder = Encoder(**ddconfig)
        self.quant_conv_ldmk = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*self.embed_dim, 1)
        self.quant_ldmk_and_conv = torch.nn.Conv2d(2*self.embed_dim, self.embed_dim, 1)
        self.cross_concat = cross_concat
        self.logit_scale = logit_scale
        self.gather_with_grad = gather_with_grad
        self.ldmk_kl_loss = ldmk_kl_loss
        self.use_head_pose = use_head_pose
        self.use_ae = use_ae
        self.sample_posterior = False if self.use_ae else True
        self.use_unet_connection = use_unet_connection
        if self.use_head_pose:
            self.headpose_ldmk_proj = torch.nn.Conv2d(2*ddconfig["in_channels"], ddconfig["in_channels"], 1)

        downs = []
        for _ in range(num_down):
            downs.append(Downsample(2*ddconfig["z_channels"], True))
            # appear_hw = appear_hw // 2
        
        ups = []
        for _ in range(num_down):
            ups.append(Upsample(ddconfig["z_channels"], True))
            # appear_hw = appear_hw * 2

        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)

        if ckpt_path is not None:
            self.first_init = first_init
            self.init_from_ckpt(ckpt_path, ignore_keys)
    
    def encode(self, x):
        if self.use_unet_connection:
            h, hs = self.encoder(x, return_hs=True)
        else:
            h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if self.use_unet_connection:
            return posterior, h, hs
        else:
            return posterior, h

    def encode_ldmk(self, x, sample_posterior=True):
        ldmk_h = self.ldmk_encoder(x)

        # down ldmk to 16*16*3 for a2e training
        # up sample back for reconstruction
        for down in self.downs:
            ldmk_h = down(ldmk_h)

        moments = self.quant_conv_ldmk(ldmk_h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            ldmk_h = posterior.sample() # 16*16*3
        else:
            ldmk_h = posterior.mode()
        ldmk_middle = ldmk_h

        for up in self.ups:
            ldmk_h = up(ldmk_h)
        return ldmk_h, ldmk_middle, posterior

    def forward(self, input, 
                ldmks=None,
                concat_input=True,
                sample_posterior=True,
                smooth_type=None):
        "ONLY USE smooth_type FOR INFERENCE!"
        if self.use_unet_connection:
            appear_posterior, appear_middle, appear_hs = self.encode(input)
        else:
            appear_posterior, appear_middle = self.encode(input)
        if sample_posterior:
            z_appear = appear_posterior.sample()
        else:
            z_appear = appear_posterior.mode()
        
        z_ldmk, ldmk_middle, ldmk_posterior = self.encode_ldmk(ldmks, sample_posterior=sample_posterior)

        if smooth_type:
            if smooth_type == "move_avg":
                z_appear = moving_average(z_appear.detach().cpu().numpy(), w=3)
                z_ldmk = moving_average(z_ldmk.detach().cpu().numpy(), w=3)
            elif smooth_type == "one_euro":
                z_appear = one_euro_smooth(z_appear.detach().cpu().numpy(), min_cutoff=0, beta=0.5)
                z_ldmk = one_euro_smooth(z_ldmk.detach().cpu().numpy(), min_cutoff=0, beta=0.5)
            else:
                raise NotImplementedError(f"Only support 'move_avg' and 'one_euro', not {smooth_type}!")
            z_appear = torch.from_numpy(z_appear).to(input.device).type(input.dtype)
            z_ldmk = torch.from_numpy(z_ldmk).to(input.device).type(input.dtype)
        
        if concat_input:
            z_appear_input, z_appear_pos = torch.chunk(z_appear, 2, dim=0)
            z_ldmk_inputs, z_ldmk_pos = torch.chunk(z_ldmk, 2, dim=0)
            if self.cross_concat:
                z = torch.cat([z_appear_pos, z_ldmk_inputs], dim=1)
            else:
                z = torch.cat([z_appear_input, z_ldmk_inputs], dim=1)
        else:
            z = torch.cat([z_appear, z_ldmk], dim=1)
        
        z = self.quant_ldmk_and_conv(z)
        if self.use_unet_connection:
            dec = self.decoder(z, hs=appear_hs)
        else:
            dec = self.decode(z)
        return dec, appear_posterior, ldmk_posterior, ldmk_middle, appear_middle

    def decode(self, z, hs=None):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, hs=hs)
        return dec

    def forward_fromldmklatent(self,
                               appear_img,
                               ldmk_middle,
                               concat_input=True,
                               sample_posterior=True):

        appear_posterior, appear_middle = self.encode(appear_img)
        if sample_posterior:
            z_appear = appear_posterior.sample()
        else:
            z_appear = appear_posterior.mode()

        # z_ldmk, ldmk_middle, ldmk_posterior = self.encode_ldmk(ldmks, sample_posterior=sample_posterior)
        for up in self.ups:
            ldmk_middle = up(ldmk_middle)
        z_ldmk = ldmk_middle

        if concat_input:
            z_appear_input, z_appear_pos = torch.chunk(z_appear, 2, dim=0)
            z_ldmk_inputs, z_ldmk_pos = torch.chunk(z_ldmk, 2, dim=0)
            if self.cross_concat:
                z = torch.cat([z_appear_pos, z_ldmk_inputs], dim=1)
            else:
                z = torch.cat([z_appear_input, z_ldmk_inputs], dim=1)
        else:
            z = torch.cat([z_appear, z_ldmk], dim=1)
        
        z = self.quant_ldmk_and_conv(z)

        dec = self.decode(z)
        return dec

    def _get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_input(self, batch, k):
        x = self._get_input(batch, "image")
        x_pos = self._get_input(batch, "positive_img")
        x_ldmk = self._get_input(batch, "ldmk_img")
        x_ldmk_pos = self._get_input(batch, "ldmk_pos_img")
        return x, x_pos, x_ldmk, x_ldmk_pos

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs, inputs_pos, ldmks, ldmks_pos = self.get_input(batch, self.image_key)
        concat_inputs = torch.cat([inputs, inputs_pos], dim=0)
        concat_ldmks = torch.cat([ldmks, ldmks_pos], dim=0)

        if self.use_head_pose:
            img_pose = batch["img_pose"].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, ldmks.shape[2], ldmks.shape[3])
            ldmk_pose = torch.cat([ldmks, img_pose], dim=1)
            ldmks = self.headpose_ldmk_proj(ldmk_pose)
        
        reconstructions, posterior, ldmk_posterior, ldmk_middle, appear_middle = self(inputs_pos, ldmks=ldmks, concat_input=False,sample_posterior=self.sample_posterior)
        if not self.ldmk_kl_loss:
            ldmk_posterior = None
        # reconstructions, posterior, ldmk_middle, appear_middle = self(concat_inputs, ldmks=concat_ldmks)
        # z_ldmk, z_ldmk_pos = torch.chunk(ldmk_middle, 2, dim=0)
        # z_appear, z_appear_pos = torch.chunk(appear_middle, 2, dim=0)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", ldmk_posterior=ldmk_posterior)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train", ldmk_posterior=ldmk_posterior)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
    
    def validation_step(self, batch, batch_idx):
        inputs, inputs_pos, ldmks, ldmks_pos = self.get_input(batch, self.image_key)
        concat_inputs = torch.cat([inputs, inputs_pos], dim=0)
        concat_ldmks = torch.cat([ldmks, ldmks_pos], dim=0)

        if self.use_head_pose:
            img_pose = batch["img_pose"].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, ldmks.shape[2], ldmks.shape[3])
            ldmk_pose = torch.cat([ldmks, img_pose], dim=1)
            ldmks = self.headpose_ldmk_proj(ldmk_pose)

        reconstructions, posterior, ldmk_posterior, _, _ = self(inputs_pos, ldmks=ldmks, concat_input=False, sample_posterior=self.sample_posterior)
        if not self.ldmk_kl_loss:
            ldmk_posterior = None
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val", ldmk_posterior=ldmk_posterior)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val", ldmk_posterior=ldmk_posterior)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        params = list(self.encoder.parameters())+ \
                 list(self.decoder.parameters())+ \
                 list(self.ldmk_encoder.parameters())+ \
                 list(self.quant_conv.parameters())+ \
                 list(self.quant_ldmk_and_conv.parameters())+ \
                 list(self.quant_conv_ldmk.parameters())+ \
                 list(self.post_quant_conv.parameters())+ \
                 list(self.ups.parameters())+ \
                 list(self.downs.parameters())
        if self.use_head_pose:
            params += list(self.headpose_ldmk_proj.parameters())

        opt_ae = torch.optim.Adam(params,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        inputs, inputs_pos, ldmks, ldmks_pos = self.get_input(batch, self.image_key)
        concat_inputs = torch.cat([inputs, inputs_pos], dim=0).to(self.device)
        concat_ldmks = torch.cat([ldmks, ldmks_pos], dim=0).to(self.device)

        if self.use_head_pose:
            img_pose = batch["img_pose"].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, ldmks.shape[2], ldmks.shape[3])
            ldmk_pose = torch.cat([ldmks, img_pose], dim=1)
            ldmks = self.headpose_ldmk_proj(ldmk_pose)

        x = inputs.to(self.device)
        if not only_inputs:
            xrec, posterior, *_ = self(inputs_pos, ldmks=ldmks, concat_input=False)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]

        if self.first_init:
            print("First init, copy the encoder parameters to ldmk_encoder.")
            old_keys = list(sd.keys())
            for k in old_keys:
                if k.startswith("encoder"):
                    new_k = k.replace("encoder", "ldmk_encoder")
                    sd[new_k] = sd[k]

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    # def contrastive_loss(self, cond, positive_cond, 
    #                      flag="motion", 
    #                      gather_with_grad=False, 
    #                      world_size=1,
    #                      rank=0,
    #                      other_cond=None):
    #     # use the positive cond as the only one positive pair
    #     # use other conds in the batch as the negative pair
    #     device = cond.device
    #     logit_scale = self.logit_scale

    #     if gather_with_grad:
    #         # gather from all gpus, current only support 1 node
    #         # size -> b*world_size, d
    #         all_cond = torch.cat(torch.distributed.nn.all_gather(cond), dim=0)
    #         all_positive_cond = torch.cat(torch.distributed.nn.all_gather(positive_cond), dim=0)
    #     b = cond.shape[0]
    #     if flag == "appearance":
    #         if gather_with_grad:
    #             logits = (cond @ all_positive_cond.T) # local loss to save memory
    #         else:
    #             logits = (cond @ positive_cond.T)
    #         logits = logit_scale * logits
    #     elif flag == "motion":
    #         if gather_with_grad:
    #             all_logits_cond = list(torch.chunk(cond @ all_cond.T, world_size, dim=1))
    #             logits_cond = all_logits_cond[rank]
    #         else:
    #             logits_cond = cond @ cond.T
    #         logits_cond_pos = (cond*positive_cond).sum(-1) # b*d -> b
    #         logits_cond_pos = torch.diag(logits_cond_pos) # b -> b, b

    #         diag_mask = torch.eye(b).to(device)
    #         logits = (1-diag_mask)*logits_cond + diag_mask*logits_cond_pos
    #         logits = logit_scale * logits
    #         if b > 5:
    #             # mask the frames in two consecutive timesteps, which should not be negative pairs
    #             mask_pos = torch.diag(torch.ones(b-1), 1).bool() | torch.diag(torch.ones(b-1), -1).bool() | \
    #                         torch.diag(torch.ones(b-2), 2).bool() | torch.diag(torch.ones(b-2), -2).bool()
    #             logits.masked_fill_(mask_pos.to(device), float('-inf'))
            
    #         if gather_with_grad:
    #             all_logits_cond[rank] = logits
    #             logits = torch.cat(all_logits_cond, dim=1)
    #     else:
    #         raise NotImplementedError()
        
    #     if self.conf.mutual_neg:
    #         logits_mutual = (cond*other_cond).sum(-1)
    #         logits_mutual = logit_scale * logits_mutual
    #         logits = torch.cat([logits, logits_mutual.unsqueeze(1)], dim=1)

    #     # logits = logit_scale * logits


    #     if gather_with_grad:
    #         if self.prev_num_logits != b or device not in self.labels:
    #             labels = torch.arange(b, device=device, dtype=torch.long)
    #             if world_size > 1:
    #                 labels = labels + b * rank
    #             self.labels[device] = labels
    #             self.prev_num_logits = b
    #         else:
    #             labels = self.labels[device]
    #     else:
    #         labels = torch.arange(b, device=device, dtype=torch.long)
        
    #     contrast_loss = F.cross_entropy(logits, labels, reduction ='none')
    #     # if self.conf.mutual_neg:
    #     #     contrast_loss = contrast_loss[:-1]
        
    #     return contrast_loss

class AutoEncoderKLWithProj(AutoEncoderKLwithLDMK):
    def __init__(self, ddconfig, first_init=False, bottleneck_dim=128, latent_noise_scale=0.0, *args, **kwargs):
       
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        self.bottle_neck_dim = bottleneck_dim
        self.latent_noise_scale = latent_noise_scale
        super().__init__(ddconfig, *args, ckpt_path=None, **kwargs) # do not load ckpt here
        self.projector = nn.Sequential(
            nn.Linear(768, self.bottle_neck_dim),
            nn.ReLU())
        self.enlarge = nn.Sequential(
            nn.Linear(self.bottle_neck_dim, 768),
            nn.ReLU())
        if ckpt_path is not None:
            self.first_init = first_init
            self.init_from_ckpt(ckpt_path, ignore_keys)

    def encode_ldmk(self, x, sample_posterior=True, noise_in_val=0.0):
        ldmk_h = self.ldmk_encoder(x)
        
        # down ldmk to 16*16*3 for a2e training
        # up sample back for reconstruction
        for down in self.downs:
            ldmk_h = down(ldmk_h)
        
        moments = self.quant_conv_ldmk(ldmk_h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            ldmk_h = posterior.sample() # 16*16*3
        else:
            ldmk_h = posterior.mode()
        ldmk_h = self.projector(ldmk_h.reshape(ldmk_h.shape[0], -1))
        ldmk_middle = ldmk_h
        if self.training and self.latent_noise_scale > 0:
            ldmk_h = ldmk_h + torch.randn_like(ldmk_h) * self.latent_noise_scale
        elif not self.training and noise_in_val > 0:
            ldmk_h = ldmk_h + torch.randn_like(ldmk_h) * noise_in_val
        ldmk_h = self.enlarge(ldmk_h).reshape(ldmk_h.shape[0], 3, 16, 16)
        for up in self.ups:
            ldmk_h = up(ldmk_h)
        return ldmk_h, ldmk_middle, posterior
    
    def forward(self, input, 
                ldmks=None,
                concat_input=True,
                sample_posterior=True,
                smooth_type=None,
                noise_in_val=0.0):
        "ONLY USE smooth_type FOR INFERENCE!"
        if self.use_unet_connection:
            appear_posterior, appear_middle, appear_hs = self.encode(input)
        else:
            appear_posterior, appear_middle = self.encode(input)
        if sample_posterior:
            z_appear = appear_posterior.sample()
        else:
            z_appear = appear_posterior.mode()
        
        z_ldmk, ldmk_middle, ldmk_posterior = self.encode_ldmk(ldmks, sample_posterior=sample_posterior, noise_in_val=noise_in_val)

        if smooth_type:
            if smooth_type == "move_avg":
                z_appear = moving_average(z_appear.detach().cpu().numpy(), w=3)
                z_ldmk = moving_average(z_ldmk.detach().cpu().numpy(), w=3)
            elif smooth_type == "one_euro":
                z_appear = one_euro_smooth(z_appear.detach().cpu().numpy(), min_cutoff=0, beta=0.5)
                z_ldmk = one_euro_smooth(z_ldmk.detach().cpu().numpy(), min_cutoff=0, beta=0.5)
            else:
                raise NotImplementedError(f"Only support 'move_avg' and 'one_euro', not {smooth_type}!")
            z_appear = torch.from_numpy(z_appear).to(input.device).type(input.dtype)
            z_ldmk = torch.from_numpy(z_ldmk).to(input.device).type(input.dtype)
        
        if concat_input:
            z_appear_input, z_appear_pos = torch.chunk(z_appear, 2, dim=0)
            z_ldmk_inputs, z_ldmk_pos = torch.chunk(z_ldmk, 2, dim=0)
            if self.cross_concat:
                z = torch.cat([z_appear_pos, z_ldmk_inputs], dim=1)
            else:
                z = torch.cat([z_appear_input, z_ldmk_inputs], dim=1)
        else:
            z = torch.cat([z_appear, z_ldmk], dim=1)
        
        z = self.quant_ldmk_and_conv(z)
        if self.use_unet_connection:
            dec = self.decoder(z, hs=appear_hs)
        else:
            dec = self.decode(z)
        return dec, appear_posterior, ldmk_posterior, ldmk_middle, appear_middle

  
    def configure_optimizers(self):
        # lr = self.learning_rate
        # TODO may be changed when all the parameters are trained
        # proj_paras = []
        # for name, p in self.named_parameters():
        #     if 'projector' in name or 'enlarge' in name:
        #         proj_paras.append(p)
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False

        # opt_ae = torch.optim.Adam(proj_paras,
        #                           lr=lr, betas=(0.5, 0.9))
        
        # for p in self.loss.discriminator.parameters():
        #     p.requires_grad = True
            
        # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
        #                             lr=lr, betas=(0.5, 0.9))

        # return [opt_ae, opt_disc], []
        lr = self.learning_rate

        params = list(self.encoder.parameters())+ \
                 list(self.decoder.parameters())+ \
                 list(self.ldmk_encoder.parameters())+ \
                 list(self.quant_conv.parameters())+ \
                 list(self.quant_ldmk_and_conv.parameters())+ \
                 list(self.quant_conv_ldmk.parameters())+ \
                 list(self.post_quant_conv.parameters())+ \
                 list(self.ups.parameters())+ \
                 list(self.downs.parameters())+\
                 list(self.projector.parameters())+\
                 list(self.enlarge.parameters())
      
        opt_ae = torch.optim.Adam(params,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]

        if self.first_init:
            print("First init, copy the encoder parameters to ldmk_encoder.")
            old_keys = list(sd.keys())
            for k in old_keys:
                if k.startswith("encoder"):
                    new_k = k.replace("encoder", "ldmk_encoder")
                    sd[new_k] = sd[k]

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")



class AutoEncoderKLWithTemporal(AutoEncoderKLwithLDMK):
    def __init__(self, ddconfig, gradient_checkpointing, temporal_kernel_size, temporal_len, first_init=False, *args, **kwargs):
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_len = temporal_len
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(ddconfig, *args, ckpt_path=None, **kwargs) # do not load ckpt here
        if gradient_checkpointing:
            self.encoder = EncoderCKPT(**ddconfig, gradient_checkpointing=gradient_checkpointing)
            self.ldmk_encoder = EncoderWithTemporalCKPT(**ddconfig, gradient_checkpointing=gradient_checkpointing, temporal_kernel_size=self.temporal_kernel_size, temporal_len=self.temporal_len)
            self.decoder = DecoderWithTemporalCKPT(**ddconfig, gradient_checkpointing=gradient_checkpointing, temporal_kernel_size=self.temporal_kernel_size, temporal_len=self.temporal_len)
            
        else:
            self.encoder = Encoder(**ddconfig)
            self.ldmk_encoder = EncoderWithTemporalCKPT(**ddconfig, gradient_checkpointing=False, temporal_kernel_size=self.temporal_kernel_size, temporal_len=self.temporal_len)
            self.decoder = DecoderWithTemporalCKPT(**ddconfig, gradient_checkpointing=False, temporal_kernel_size=self.temporal_kernel_size, temporal_len=self.temporal_len, )
        if ckpt_path is not None:
            self.first_init = first_init
            self.init_from_ckpt(ckpt_path, ignore_keys)

    # merge bsz and time axis for spatio conv
    def get_input(self, batch, k):
        x = self._get_input(batch, "image")
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x_pos = self._get_input(batch, "positive_img")
        x_pos = x_pos.reshape(-1, x_pos.shape[2], x_pos.shape[3], x_pos.shape[4])
        x_ldmk = self._get_input(batch, "ldmk_img")
        x_ldmk = x_ldmk.reshape(-1, x_ldmk.shape[2], x_ldmk.shape[3], x_ldmk.shape[4])
        x_ldmk_pos = self._get_input(batch, "ldmk_pos_img")
        x_ldmk_pos = x_ldmk_pos.reshape(-1, x_ldmk_pos.shape[2], x_ldmk_pos.shape[3], x_ldmk_pos.shape[4])
        return x, x_pos, x_ldmk, x_ldmk_pos

    def configure_optimizers(self):
        lr = self.learning_rate
        # TODO may be changed when all the parameters are trained
        temporal_paras = []
        for name, p in self.named_parameters():
            if 'temp' in name:
                temporal_paras.append(p)
                p.requires_grad = True
            else:
                p.requires_grad = False

        opt_ae = torch.optim.Adam(temporal_paras,
                                  lr=lr, betas=(0.5, 0.9))
        
        for p in self.loss.discriminator.parameters():
            p.requires_grad = True
            
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        return [opt_ae, opt_disc], []


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]

        if self.first_init:
            print("First init, copy the encoder parameters to ldmk_encoder.")
            old_keys = list(sd.keys())
            for k in old_keys:
                if k.startswith("encoder"):
                    new_k = k.replace("encoder", "ldmk_encoder")
                    sd[new_k] = sd[k]

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


class AutoEncoderKLforContrast(AutoencoderKL):
    def __init__(self,
                cond_stage_config, 
                cond_stage_key="text", # image or audio
                cond_stage_trainable=False,
                contrast_config_path=None,
                middle_hw=64,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key

        self.instantiate_cond_stage(cond_stage_config)

        contrast_config = CLIPVisionConfig.from_pretrained(contrast_config_path)
        self.contrast_head = VisAudioContrastHead(config=contrast_config,
                                                  middle_hw=middle_hw)
    
    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model
    
    def encode(self, x):
        h = self.encoder(x)
        assert isinstance(h, tuple), "Encoder must return a tuple of (final_out, middle_out)"
        h, middle_out = h
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, middle_out

    def forward(self, input, sample_posterior=True):
        posterior, middle_out = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)

        _, pooled_contrast_out, logit_scale = self.contrast_head(middle_out)
        return dec, posterior, pooled_contrast_out, logit_scale
    
    def get_input(self, batch, k, cond_key="text"):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

        y = batch[cond_key]
        assert not self.cond_stage_trainable, "Trainable condition model not implemented yet."
        y = list(y[0])
        y = self.cond_stage_model(y, return_pooled=True)
        return x, y
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.contrast_head.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs, conds_hidden = self.get_input(batch, self.image_key)
        # conds_hidden: (bsz, hidden_size=768), middle_out: (bsz, c=512, h=64, w=64)
        # downsample and transfer middle_out to (bsz, 16*16, 768) 
        reconstructions, posterior, contrast_out, logit_scale = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            cond_hidden=conds_hidden, contrast_hidden=contrast_out, logit_scale=logit_scale)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train",
                                                cond_hidden=conds_hidden, contrast_hidden=contrast_out, logit_scale=logit_scale)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs, conds_hidden = self.get_input(batch, self.image_key)
        reconstructions, posterior, contrast_out, logit_scale = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val",
                                        cond_hidden=conds_hidden, contrast_hidden=contrast_out, logit_scale=logit_scale)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val",
                                            cond_hidden=conds_hidden, contrast_hidden=contrast_out, logit_scale=logit_scale)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x, _ = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior, contrast_out, logit_scale = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log


class VisAudioContrastHead(nn.Module):
    """
    Add a contrastive head on the encoder of the vae.
    The vae is fine-tuned from StableDiffusion at this time. Use the middle output as the image representation, size: (bsz, 512, 64, 64)
    Then use two conv layers to reduce the size to (bsz, 512, 16, 16) then reshape to (bsz, 256, 512).
    """
    def __init__(self, 
                 config, # CLIPVisionConfig
                 resamp_with_conv=True,
                 num_downsamples=2,
                 middle_hw=64,
                 num_res_blocks=1,
                 block_ch=512,
                 dropout=0.0,):
        super().__init__()
        self.config = config
        self.num_downsamples = num_downsamples
        self.num_res_blocks = num_res_blocks
        output_ch = config.hidden_size

        self.down = nn.ModuleList()
        for i_level in range(num_downsamples):
            block = nn.ModuleList()
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_ch,
                                         out_channels=block_ch,
                                         temb_channels=0,
                                         dropout=dropout))
            down = nn.Module()
            down.block = block
            down.downsample = Downsample(block_ch, resamp_with_conv)
            self.down.append(down)
        
        max_position_embeddings = (middle_hw // (2 ** num_downsamples)) ** 2 + 1

        self.out_projection = None
        if output_ch != block_ch:
            self.out_projection = nn.Linear(block_ch, output_ch, bias=False)

        layer_norm_eps = getattr(config, "layer_norm_eps", 1e-5)
        self.contras_encoder = CLIPEncoder(config)

        self.class_embedding = nn.Parameter(torch.randn(output_ch))
        self.pre_layrnorm = nn.LayerNorm(output_ch, eps=layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(output_ch, eps=layer_norm_eps)
        self.position_embedding = nn.Embedding(max_position_embeddings, output_ch)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, 
                hidden_states,
                add_position_emb=True,
        ):
        """
        hidden_states: the middle output of the vae encoder, size: (bsz, 512, 64, 64)
                       downsample and project to (bsz, 256, 768)
        output: (bsz, 768)
        """
        bsz, seq_len, _, _ = hidden_states.size()

        # downsample. from (bsz, 512, 64, 64) to (bsz, 512, 16, 16)
        h = hidden_states
        for i_level in range(self.num_downsamples):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, None)

            h = self.down[i_level].downsample(h)
        
        hidden_states = h.flatten(2).transpose(1, 2) # (bsz, 256, 512)

        if self.out_projection is not None:
            hidden_states = self.out_projection(hidden_states) # (bsz, 256, 768)

        class_embeds = self.class_embedding.expand(bsz, 1, -1)
        hidden_states = torch.cat([class_embeds, hidden_states], dim=1)
        seq_len += 1
            
        if add_position_emb:
            position_ids = self.position_ids[:, :seq_len]
            position_embeddings = self.position_embedding(position_ids)
            hidden_states = hidden_states + position_embeddings

        encoder_out = self.contras_encoder(
            inputs_embeds=hidden_states,
            return_dict=True
        )

        last_hidden_state = encoder_out.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.final_layer_norm(pooled_output)

        return last_hidden_state, pooled_output, self.logit_scale.exp()

class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x