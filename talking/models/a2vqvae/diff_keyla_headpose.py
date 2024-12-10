#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import transformers
import random

from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from lightning.pytorch.utilities import rank_zero_info
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

from ldm.util import count_params, instantiate_from_config, instantiate_lrscheduler_from_config
from ldm.modules.ema import LitEma
from talking.util import make_non_pad_mask, _paraphrase, all_paraphrase, add_au_func, action_units_dict, get_au_label, add_au_func_para
from talking.util import all_paraphrase_extend, all_paraphrase_intensity
from talking.modules.a2vqvae.clip import FrozenCLIPEmbedder
from talking.modules.a2vqvae.clip_adapter import Adapter
from talking.modules.a2vqvae.style_encoder import StyleEncoder



class TalkingA2VqvaeKeylaPoseModel(pl.LightningModule):
    def __init__(self,
                 backbone_config,
                 keylatent_encoder_config,
                 audio_encoder_config,
                 pose_dim=3,
                 val_with_extra_pose_npy_path=None,
                 lr_scheduler_config=None,
                 audio_feat_dim=512,
                 loss_scale_by_dim=True,
                 diffusion_loss_type='l1',
                 diffusion_loss_pose_weight=1.0,
                 diffusion_loss_x0_weight=1.0,
                 diffusion_loss_noise_weight=1.0,
                 key_index_weight=1.0,
                 zero_optim=False,
                 noise_factor=1.0,
                 offset=1e-5,
                 beta_min=0.05,
                 beta_max=20.0,
                 weight_decay=0.0,
                 latent_denorm_mean=None,
                 latent_denorm_std=None,
                 keyframe_same_in_batch=True,
                 ckpt_path=None,
                 ckpt_path2=None,
                 print_backbone=False,
                 use_ema=False,
                 ignore_keys=[],
                 monitor=None
                 ):
        super().__init__()
        # Verify the paths
        ckpt_path = self._verify_paths(ckpt_path, ckpt_path2)
        self.loss_scale_by_dim = loss_scale_by_dim
        self.diffusion_loss_type = diffusion_loss_type
        self.diffusion_loss_pose_weight = diffusion_loss_pose_weight
        self.diffusion_loss_x0_weight = diffusion_loss_x0_weight
        self.diffusion_loss_noise_weight = diffusion_loss_noise_weight
        self.key_index_weight = key_index_weight
        self.zero_optim = zero_optim
        self.noise_factor = noise_factor
        self.offset = offset
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.weight_decay = weight_decay
        self.latent_denorm_mean = latent_denorm_mean
        self.latent_denorm_std = latent_denorm_std
        self.keyframe_same_in_batch = keyframe_same_in_batch
        self.model_dim = backbone_config.params.out_dims
        self.middle_dim = backbone_config.params.d_model

        self.use_scheduler = lr_scheduler_config is not None
        if self.use_scheduler:
            self.lr_scheduler_config = lr_scheduler_config

        self.val_with_extra_pose_npy_path = val_with_extra_pose_npy_path
        self.pose_pred = ConditionPredictor(audio_encoder_config.params.output_dim)
        self.pose_proj = torch.nn.Linear(pose_dim, audio_encoder_config.params.output_dim)

        # init models
        self.keylatent_encoder = instantiate_from_config(keylatent_encoder_config)
        # https://pytorch.org/audio/0.13.1/generated/torchaudio.models.Conformer.html#torchaudio.models.Conformer
        self.audio_encoder = instantiate_from_config(audio_encoder_config)
        self.audio_downsample = nn.Conv1d(audio_feat_dim, audio_encoder_config.params.input_dim, kernel_size=3, stride=2, padding=1)
        # conformer as default
        self.backbone = instantiate_from_config(backbone_config)

        # prepare model
        count_params(self.backbone, verbose=True, verbose_prefix='[TalkingA2VqvaeKeylaPoseModel] ')
        self.use_ema = use_ema
        if self.use_ema:
            self.backbone_ema = LitEma(self.backbone)
            rank_zero_info(f"[TalkingA2VqvaeKeylaPoseModel] Keeping EMAs of {len(list(self.backbone_ema.buffers()))}.")

        if print_backbone:
            rank_zero_info(self.backbone)

        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            self._init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def _init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    rank_zero_info("[TalkingA2VqvaeKeylaPoseModel] Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        rank_zero_info(f"[TalkingA2VqvaeKeylaPoseModel] Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            rank_zero_info(f"[TalkingA2VqvaeKeylaPoseModel] Missing Keys: {missing}")
            rank_zero_info(f"[TalkingA2VqvaeKeylaPoseModel] Unexpected Keys: {unexpected}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.backbone_ema.store(self.backbone.parameters())
            self.backbone_ema.copy_to(self.backbone)
            if context is not None:
                rank_zero_info(f"[TalkingA2VqvaeKeylaPoseModel] {context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.backbone_ema.restore(self.backbone.parameters())
                if context is not None:
                    rank_zero_info(f"[TalkingA2VqvaeKeylaPoseModel] {context}: Restored training weights")

    @torch.no_grad()
    def _get_noise(self, t, beta_init, beta_term, cumulative=False):
        if cumulative:
            noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
        else:
            noise = beta_init + (beta_term - beta_init)*t
        return noise

    def get_input(self, batch):
        latent = batch['latent'].to(memory_format=torch.contiguous_format)
        audio_feat = batch['audio_feat'].to(memory_format=torch.contiguous_format)
        keylatent = batch['keylatent'].to(memory_format=torch.contiguous_format)
        pose = batch['headpose'].to(memory_format=torch.contiguous_format)
        # get mask
        masks = make_non_pad_mask(batch['latent_length'], xs=batch['latent'][..., 0])
        masks = masks.to(memory_format=torch.contiguous_format)  #  (bs, max_num_frames)
        masks = rearrange(masks, 'b t -> b t 1').contiguous()
        # masks: (bs, T, 1)
        return audio_feat, keylatent, pose, latent, masks

    def get_learned_conditioning(self, audio_feat, keylatent, masks, pose=None):
        # audio_feat: (bs, T*2, 512)
        # keylatent: (bs, T, 512)
        # pose: (bs, T, 3)

        # get keyframe latent
        keylatent_feat = self.keylatent_encoder(keylatent)
        # keylatent_feat: (bs, T, d)

        # get downsampled audio feat
        audio_feat = self.audio_downsample(audio_feat.transpose(1, 2)).transpose(1, 2)  # (bs, T, 512)
        audio_feat = self.audio_encoder(audio_feat, masks=masks)
        # get keyframe latent
        keylatent_feat = self.keylatent_encoder(keylatent)
        # pose prediction
        out_pose = self.pose_pred(audio_feat, ~masks)  # (bs, T, 3)
        if pose is not None:
            audio_feat += self.pose_proj(pose)
        else:
            audio_feat += self.pose_proj(out_pose)  # (bs, T, d)

        return audio_feat, keylatent_feat, out_pose

    def forward_diffusion(self, x0, t, mu=0.0):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance) * self.noise_factor
        return xt, z

    @torch.no_grad()
    def reverse_diffusion(self, z, cond1, cond2, n_timesteps, mu=0.0, masks=None):
        h = 1.0 / n_timesteps
        xt = z
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = self._get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            # xt: (bs, T, 512)
            # t: (bs)
            # cond: (bs, T, 512)
            x0_ = self.backbone(input=xt,
                                diffusion_step=t,
                                cond1=cond1,
                                cond2=cond2,
                                masks=masks)

            cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
            rho = x0_*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
            noise_pred = xt - rho
            lambda_ = 1.0 - torch.exp(-cum_noise)
            logp = - noise_pred / (lambda_ + 1e-8)
            dxt = 0.5 * (mu - xt - logp)
            dxt = dxt * noise_t * h
            xt = xt - dxt
        return xt

    @torch.no_grad()
    def sample(self, z_shape, cond1, cond2, n_timesteps=150, temperature=1.8, mu=0.0, masks=None):
        # z_shape: latent.shape
        z = mu + torch.randn(z_shape, dtype=cond1.dtype, device=cond1.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.reverse_diffusion(z=z, cond1=cond1, cond2=cond2, n_timesteps=n_timesteps, mu=mu, masks=masks)
        return decoder_outputs

    def _weights_nonzero(self, target, dtype):
        # target : bs x T x d
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).to(dtype=dtype).repeat(1, 1, dim)

    def get_loss(self, pred, target, masks):
        if self.loss_scale_by_dim:
            masks = masks.to(dtype=pred.dtype).repeat(1, 1, pred.size(-1))
        else:
            masks = masks.to(dtype=pred.dtype)
        if self.key_index_weight != 1.0:
            orig_sum = masks.sum()
            masks.index_fill_(-1, torch.tensor(self._get_mouth_related_ldmk_index(), device=pred.device), self.key_index_weight)
            masks = masks / masks.sum() * orig_sum
        # diffusion loss
        if self.diffusion_loss_type == "l1":
            # pred : bs x T x d
            # target : bs x T x d
            l1_loss = F.l1_loss(pred, target, reduction='none')
            # masks = self._weights_nonzero(target, l1_loss.dtype)
            loss = (l1_loss * masks).sum() / masks.sum()
        elif self.diffusion_loss_type == "l2":
            # pred : bs x T x d
            # target : bs x T x d
            mse_loss = F.mse_loss(pred, target, reduction='none')
            loss = (mse_loss * masks).sum() / masks.sum()
        else:
            raise ValueError(f"[TalkingA2VqvaeKeylaPoseModel] Unknown diffusion loss type: {self.diffusion_loss_type}")
        return loss

    def get_pose_loss(self, pred, target, masks):
        if self.loss_scale_by_dim:
            masks = masks.to(dtype=pred.dtype).repeat(1, 1, pred.size(-1))
        else:
            masks = masks.to(dtype=pred.dtype)
        mse_loss = F.mse_loss(pred, target, reduction='none')
        masks = self._weights_nonzero(target, mse_loss.dtype)
        loss = (mse_loss * masks).sum() / masks.sum()
        return loss

    def forward(self, latent, cond1, cond2, mu=0.0, masks=None):
        t = torch.rand(latent.shape[0], dtype=latent.dtype, device=latent.device,
                       requires_grad=False)
        t = torch.clamp(t, self.offset, 1.0 - self.offset)

        # get xt and noise
        xt, z = self.forward_diffusion(latent, t, mu)  # xt, z: (bs, T, 512)

        x0_estimation = self.backbone(input=xt,
                                      diffusion_step=t,
                                      cond1=cond1,
                                      cond2=cond2,
                                      masks=masks)
        # x0_estimation: (bs, T, 512)

        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        rho = x0_estimation * torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        lambda_ = 1.0 - torch.exp(-cum_noise)

        noise_pred = (xt - rho) / (torch.sqrt(lambda_))
        return x0_estimation, noise_pred, z

    def training_step(self, batch, batch_idx):
        """
        batch keys:
        latent: (bs, T, 512)
        audio_feat: (bs, T*2, 512)
        keylatent: (bs, T, 512)
        """
        audio_feat, keylatent, pose, latent, masks = self.get_input(batch)
        audio_feat, keylatent_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, pose=pose)
        # audio_feat: (bs, T, 512)
        # keylatent_feat: (bs, T, 512)
        x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, masks=masks)

        # log to lightning
        loss_pose = self.get_pose_loss(pose_pred, pose, masks)
        loss_x0 = self.get_loss(x0_pred, latent, masks)
        loss_noise = self.get_loss(noise_pred, noise_gt, masks)

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_pose': loss_pose.mean() * self.diffusion_loss_pose_weight})
        loss_dict.update({f'{log_prefix}/loss_x0': loss_x0.mean() * self.diffusion_loss_x0_weight})
        loss_dict.update({f'{log_prefix}/loss_noise': loss_noise * self.diffusion_loss_noise_weight})

        loss = self.diffusion_loss_pose_weight * loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise
        loss_dict.update({f'{log_prefix}/loss': loss})

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            # self.log(f'{log_prefix}/lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # log to tensorboard
        # https://www.pytorchlightning.ai/blog/tensorboard-with-pytorch-lightning
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_pose', loss_pose.mean(), self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_x0', loss_x0.mean(), self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_noise', loss_noise, self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss', loss, self.global_step)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        audio_feat, keylatent, pose, latent, masks = self.get_input(batch)
        audio_feat, keylatent_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks)
        # audio_feat: (bs, T, 512)
        # keylatent_feat: (bs, T, 512)
        x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, masks=masks)

        loss_pose = self.get_pose_loss(pose_pred, pose, masks)
        loss_x0 = self.get_loss(x0_pred, latent, masks)
        loss_noise = self.get_loss(noise_pred, noise_gt, masks)

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_pose': loss_pose.mean() * self.diffusion_loss_pose_weight})
        loss_dict.update({f'{log_prefix}/loss_x0': loss_x0.mean() * self.diffusion_loss_x0_weight})
        loss_dict.update({f'{log_prefix}/loss_noise': loss_noise * self.diffusion_loss_noise_weight})

        loss = self.diffusion_loss_pose_weight * loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise
        loss_dict.update({f'{log_prefix}/loss': loss})

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    # def lr_scheduler_step(self, scheduler, metric):
    #     """Neccessary for timm lr scheduler. Not compatiable with pl1.9
    #     """
    #     if 'timm.scheduler' in self.lr_scheduler_config.target:
    #         scheduler.step(epoch=self.current_epoch)
    #     else:
    #         if metric is None:
    #             scheduler.step()
    #         else:
    #             scheduler.step(metric)

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.zero_optim:
            opt = ZeroRedundancyOptimizer(self.parameters(),
                                          optimizer_class=torch.optim.AdamW,
                                          lr=lr,
                                          betas=(0.9, 0.98),
                                          weight_decay=self.weight_decay)
        else:
            opt = torch.optim.AdamW(self.parameters(),
                                    lr=lr,
                                    betas=(0.9, 0.98),
                                    weight_decay=self.weight_decay)
        # https://lightning.ai/docs/pytorch/stable/common/optimization.html#bring-your-own-custom-learning-rate-schedulers
        if self.use_scheduler:
            rank_zero_info(f"[TalkingA2VqvaeKeylaPoseModel] Use lr scheduler: {self.lr_scheduler_config.target}")
            if 'inverse_sqrt' in self.lr_scheduler_config.target:
                scheduler = transformers.get_inverse_sqrt_schedule(optimizer=opt, num_warmup_steps=self.lr_scheduler_config.params.num_warmup_steps)
                # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
                lr_scheduler = {
                    'scheduler': scheduler,
                    'name': 'main-LR',
                    'interval': 'step',
                    'frequency': self.lr_scheduler_config.params.frequency if hasattr(self.lr_scheduler_config.params, 'frequency') else 1
                    }
            else:
                lr_scheduler = instantiate_lrscheduler_from_config(opt, self.lr_scheduler_config)
            return [opt], [lr_scheduler]
        return opt

    def _verify_paths(self, path1, path2):
        if (path1 is not None) and os.path.isfile(path1):
            return path1
        elif (path2 is not None) and os.path.isfile(path2):
            return path2
        else:
            rank_zero_info(f"[TalkingA2VqvaeKeylaPoseModel] Invalid paths {path1}, {path2}")

    def _denormalize_latent(self, latent):
        if (self.latent_denorm_mean is not None) and (self.latent_denorm_std is not None):
            latent = latent * self.latent_denorm_std + self.latent_denorm_mean
        return latent

    def _get_mouth_related_ldmk_index(self):
        lip_index = [535, 379, 380, 384, 636, 69, 65, 64, 221] + [226, 219, 224, 74, 538, 539, 385, 258]
        mouth_edge_index = [657, 1, 204, 209, 218, 60, 215, 44, 57, 216, 203, 639, 517, 530, 372, 359, 529, 375, 532, 523, 518, 316, 657]
        return lip_index + mouth_edge_index

    @torch.no_grad()
    def evaluate(self, audio_feat, keylatent, diff_init_shape, pose=None):
        """evaluate a2e from the given audio_feat and keylatent
        audio_feat: (bs, T*2, 512)
        keylatent: (bs, T, 768)
        """
        # get keyframe latent (bs, T, 768)
        keylatent = keylatent.repeat(1, audio_feat.shape[1]//2, 1)

        # get mask
        latent_length = torch.tensor([audio_feat.shape[1]//2]).to(audio_feat.device)
        masks = make_non_pad_mask(latent_length, xs=keylatent[..., 0]).to(memory_format=torch.contiguous_format)
        masks = rearrange(masks, 'b t -> b t 1').contiguous()
        
        audio_feat, keylatent_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, pose=pose)

        if isinstance(diff_init_shape, int):
            diff_init_shape = (audio_feat.shape[0], audio_feat.shape[1], diff_init_shape)
        latent_rec = self.sample(diff_init_shape, cond1=audio_feat, cond2=keylatent_feat, masks=masks)
        latent_rec = self._denormalize_latent(latent_rec)
        return latent_rec

    @torch.no_grad()
    def log_latents(self, batch, **kwargs):
        log = dict()
        audio_feat, keylatent, pose, latent, masks = self.get_input(batch)
        audio_feat, keylatent_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks)
        # if self.val_with_extra_pose_npy_path is not None:
        #     # '/data/datasets/HDTF/clean_videos_split10s_ptcode/head_pose/RD_Radio10_000-00000.npy'
        #     # '/data/datasets/HDTF/clean_videos_split10s_ptcode/head_pose/RD_Radio11_000-00000.npy'
        #     rank_zero_info(f"[TalkingA2VqvaeKeylaPoseModel] Use extra pose: {self.val_with_extra_pose_npy_path}")
        #     extra_pose = np.load(self.val_with_extra_pose_npy_path)
        #     extra_pose = torch.from_numpy(extra_pose).float().to(pose.device)[:pose.shape[1], :].unsqueeze(0)
        #     audio_feat, keylatent_feat, out_pose = self.get_learned_conditioning(audio_feat, keylatent, masks, pose=extra_pose)
        # audio_feat: (bs, T, 512)
        # keylatent_feat: (bs, T, 512)
        latent_rec = self.sample(latent.shape, cond1=audio_feat, cond2=keylatent_feat, masks=masks)
        log["latent"] = self._denormalize_latent(latent)
        log["latent_rec"] = self._denormalize_latent(latent_rec)
        log["latent_filename"] = batch['latent_filename']
        return log


class ConditionPredictor(torch.nn.Module):
    def __init__(
        self, idim, keyla_dim=0, pose_dim=3, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1
    ):
        """Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super().__init__()
        self.conv = nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim + keyla_dim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = nn.Linear(n_chans, pose_dim)

    def forward(self, xs, x_masks=None, keyla=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input audio_feat (bs, T, d).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).
        """
        xs = xs.detach()
        if keyla is not None:
            keyla = keyla.detach()
            xs = torch.cat([xs, keyla], dim=-1)
        xs = xs.transpose(1, 2).clone()  # torch.Size([bs, d, T])
        for f in self.conv:
            xs = f(xs)  # (bs, 384, T)

        xs = self.linear(xs.transpose(1, 2))  # (bs, T, 3)

        # xs: (bs, T, 3)
        # x_masks: (bs, T, 1)
        if x_masks is not None:
            # replace xs with zero where mask == True
            xs = xs.masked_fill(x_masks, 0.0)
        return xs
    
    
class TETFTalkingA2VqvaeKeylaPoseModel(TalkingA2VqvaeKeylaPoseModel):
    def __init__(self, text_encoder_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instruct_encoder = instantiate_from_config(text_encoder_config)
        self._paraphrase = _paraphrase
        self.clip_text_encoder = FrozenCLIPEmbedder()
        
    def get_input(self, batch):
        latent = batch['latent'].to(memory_format=torch.contiguous_format)
        audio_feat = batch['audio_feat'].to(memory_format=torch.contiguous_format)
        keylatent = batch['keylatent'].to(memory_format=torch.contiguous_format)
        pose = batch['headpose'].to(memory_format=torch.contiguous_format)
        instruct_sen = list(map(self._paraphrase, batch['instructs']))
        instruct_feat = self.clip_text_encoder.encode(instruct_sen).to(memory_format=torch.contiguous_format)
        # get mask
        masks = make_non_pad_mask(batch['latent_length'], xs=batch['latent'][..., 0])
        masks = masks.to(memory_format=torch.contiguous_format)  #  (bs, max_num_frames)
        masks = rearrange(masks, 'b t -> b t 1').contiguous()
        # masks: (bs, T, 1)
        return audio_feat, keylatent, latent, masks, instruct_feat, pose

    def get_learned_conditioning(self, audio_feat, keylatent, masks, instructions, pose=None):
        # audio_feat: (bs, T*2, 512)
        # keylatent: (bs, T, 512)
        # get downsampled audio feat
        
        audio_feat = self.audio_downsample(audio_feat.transpose(1, 2)).transpose(1, 2)
        # audio_feat: (bs, T, 512)
        # audio encoder
        audio_feat = self.audio_encoder(audio_feat, masks=masks)
        # get keyframe latent
        keylatent_feat = self.keylatent_encoder(keylatent)
        # keylatent_feat: (bs, T, 512)
        
        inst_feat = self.instruct_encoder(instructions)
        out_pose = self.pose_pred(audio_feat, ~masks)  # (bs, T, 3)
        if pose is not None:
            audio_feat += self.pose_proj(pose)
        else:
            audio_feat += self.pose_proj(out_pose)  # (bs, T, d)
        
        return audio_feat, keylatent_feat, inst_feat, out_pose

   

    def forward(self, latent, cond1, cond2, cond3, mu=0.0, masks=None):
        t = torch.rand(latent.shape[0], dtype=latent.dtype, device=latent.device,
                       requires_grad=False)
        t = torch.clamp(t, self.offset, 1.0 - self.offset)

        # get xt and noise
        xt, z = self.forward_diffusion(latent, t, mu)  # xt, z: (bs, T, 512)

        x0_estimation = self.backbone(input=xt,
                                      diffusion_step=t,
                                      cond1=cond1,
                                      cond2=cond2,
                                      cond3=cond3,
                                      masks=masks)
        # x0_estimation: (bs, T, 512)

        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        rho = x0_estimation * torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        lambda_ = 1.0 - torch.exp(-cum_noise)

        noise_pred = (xt - rho) / (torch.sqrt(lambda_))
        return x0_estimation, noise_pred, z



    def training_step(self, batch, batch_idx):
        """
        batch keys:
        latent: (bs, T, 512)
        audio_feat: (bs, T*2, 512)
        keylatent: (bs, T, 512)
        """
        audio_feat, keylatent, latent, masks, inst_feat, pose = self.get_input(batch)
        audio_feat, keylatent_feat, inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat, pose=pose)
        # audio_feat: (bs, T, 512)
        # keylatent_feat: (bs, T, 512)
        x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks)

        # log to lightning
        loss_pose = self.get_pose_loss(pose_pred, pose, masks)
        loss_x0 = self.get_loss(x0_pred, latent, masks)
        loss_noise = self.get_loss(noise_pred, noise_gt, masks)
        # loss_smooth = self.get_loss(x0_pred[:, 1:], x0_pred[:, :-1], masks[:, 1:]) if self.loss_smooth_weight > 0.0 else 0.0

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_pose': loss_pose.mean() * self.diffusion_loss_pose_weight})
        loss_dict.update({f'{log_prefix}/loss_x0': loss_x0.mean()})
        loss_dict.update({f'{log_prefix}/loss_noise': loss_noise})

        loss = self.diffusion_loss_noise_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise
        loss_dict.update({f'{log_prefix}/loss': loss})

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            # self.log(f'{log_prefix}/lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # log to tensorboard
        # https://www.pytorchlightning.ai/blog/tensorboard-with-pytorch-lightning
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_pose', loss_pose.mean(), self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_x0', loss_x0.mean(), self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_noise', loss_noise, self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss', loss, self.global_step)
        # self.logger.experiment.add_scalar(f'{log_prefix}/loss_smooth', loss_smooth, self.global_step)
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        audio_feat, keylatent,latent, masks, inst_feat, pose = self.get_input(batch)
        audio_feat, keylatent_feat, inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat, pose=pose)
        # audio_feat: (bs, T, 512)
        # keylatent_feat: (bs, T, 512)
        x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks)

        loss_pose = self.get_pose_loss(pose_pred, pose, masks)
        loss_x0 = self.get_loss(x0_pred, latent, masks)
        loss_noise = self.get_loss(noise_pred, noise_gt, masks)

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_pose': loss_pose.mean() * self.diffusion_loss_pose_weight})
        loss_dict.update({f'{log_prefix}/loss_x0': loss_x0.mean() * self.diffusion_loss_x0_weight})
        loss_dict.update({f'{log_prefix}/loss_noise': loss_noise * self.diffusion_loss_noise_weight})

        loss = self.diffusion_loss_pose_weight * loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise
        loss_dict.update({f'{log_prefix}/loss': loss})

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        
    @torch.no_grad()
    def reverse_diffusion(self, z, cond1, cond2, cond3, n_timesteps, mu=0.0, masks=None):
        h = 1.0 / n_timesteps
        xt = z
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = self._get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            # xt: (bs, T, 512)
            # t: (bs)
            # cond: (bs, T, 512)
            x0_ = self.backbone(input=xt,
                                diffusion_step=t,
                                cond1=cond1,
                                cond2=cond2,
                                cond3=cond3,
                                masks=masks)

            cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
            rho = x0_*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
            noise_pred = xt - rho
            lambda_ = 1.0 - torch.exp(-cum_noise)
            logp = - noise_pred / (lambda_ + 1e-8)
            dxt = 0.5 * (mu - xt - logp)
            dxt = dxt * noise_t * h
            xt = xt - dxt
        return xt

    @torch.no_grad()
    def sample(self, z_shape, cond1, cond2,cond3, n_timesteps=150, temperature=1.8, mu=0.0, masks=None):
        # z_shape: latent.shape
        z = mu + torch.randn(z_shape, dtype=cond1.dtype, device=cond1.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.reverse_diffusion(z=z, cond1=cond1, cond2=cond2, cond3=cond3, n_timesteps=n_timesteps, mu=mu, masks=masks)
        return decoder_outputs


    

    @torch.no_grad()
    def evaluate(self, audio_feat, keylatent, diff_init_shape, inst_feat, pose=None):
        """evaluate a2e from the given audio_feat and keylatent
        audio_feat: (bs, T*2, 512)
        keylatent: (bs, T, 768)
        """
        # get keyframe latent (bs, T, 768)
        keylatent = keylatent.repeat(1, audio_feat.shape[1]//2, 1)

        # get mask
        latent_length = torch.tensor([audio_feat.shape[1]//2]).to(audio_feat.device)
        masks = make_non_pad_mask(latent_length, xs=keylatent[..., 0]).to(memory_format=torch.contiguous_format)
        masks = rearrange(masks, 'b t -> b t 1').contiguous()
        
        audio_feat, keylatent_feat, inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat, pose=pose)
        if isinstance(diff_init_shape, int):
            diff_init_shape = (audio_feat.shape[0], audio_feat.shape[1], diff_init_shape)
        latent_rec = self.sample(diff_init_shape, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks)
        latent_rec = self._denormalize_latent(latent_rec)
        return latent_rec

    
    @torch.no_grad()
    def log_latents(self, batch, **kwargs):
        log = dict()
        audio_feat, keylatent, latent, masks, inst_feat, pose = self.get_input(batch)
        audio_feat, keylatent_feat,  inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat)
        # if self.val_with_extra_pose_npy_path is not None:
        #     # '/data/datasets/HDTF/clean_videos_split10s_ptcode/head_pose/RD_Radio10_000-00000.npy'
        #     # '/data/datasets/HDTF/clean_videos_split10s_ptcode/head_pose/RD_Radio11_000-00000.npy'
        #     rank_zero_info(f"[TalkingA2VqvaeKeylaPoseModel] Use extra pose: {self.val_with_extra_pose_npy_path}")
        #     extra_pose = np.load(self.val_with_extra_pose_npy_path)
        #     extra_pose = torch.from_numpy(extra_pose).float().to(pose.device)[:pose.shape[1], :].unsqueeze(0)
        #     audio_feat, keylatent_feat, out_pose = self.get_learned_conditioning(audio_feat, keylatent, masks, pose=extra_pose)
        # audio_feat: (bs, T, 512)
        # keylatent_feat: (bs, T, 512)
        latent_rec = self.sample(latent.shape, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks)
        log["latent"] = self._denormalize_latent(latent)
        log["latent_rec"] = self._denormalize_latent(latent_rec)
        log["latent_filename"] = batch['latent_filename']
        return log
    
    
    
class TETFAllTalkingA2VqvaeKeylaPoseModel(TETFTalkingA2VqvaeKeylaPoseModel):
    def __init__(self, 
                 only_update_text=False, 
                 update_text_and_q=False, 
                 inst_use_parapharse=True, 
                 use_au=False,
                 use_au_class_loss=False, 
                 au_class_loss_weight=0.0,
                 minor_init_text_cross_attn=False,
                 *args, **kwargs):
        self.update_text_and_q = update_text_and_q
        super().__init__(*args, **kwargs)
        self.all_paraphrase = all_paraphrase
        self.only_update_text = only_update_text
        self.inst_use_parapharse = inst_use_parapharse
        self.use_au = use_au
        self.add_au_func = add_au_func
        self.add_au_func_para = add_au_func_para
        self.use_au_class_loss = use_au_class_loss
        self.au_class_loss_weight = au_class_loss_weight
        if self.use_au_class_loss:
            action_units_num = len(action_units_dict.keys()) + 1
            self.au_classifier = nn.Sequential(
                nn.Linear(self.model_dim, 256),
                nn.ReLU(),
                nn.Linear(256, action_units_num),
            )
        self.minor_init_text_cross_attn = minor_init_text_cross_attn
        if self.minor_init_text_cross_attn:
            for name, param in self.named_parameters():
                if 'cross_attn_text' in name:
                    param.data.normal_(mean=0.0, std=1.0e-3)
                    param.requires_grad = True
        
        
    def _init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    rank_zero_info("[TETFAllTalkingA2VqvaeKeylaPoseModel] Deleting key {} from state_dict.".format(k))
                    del sd[k]
            if self.update_text_and_q:
                if k.startswith('backbone.conformer_layers') and k.endswith('in_proj_weight'):
                    # print(k)
                    weight = sd[k]
                    q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
                    sd[k.replace('in_proj_weight','q_proj_weight' )] = q_weight
                    sd[k.replace('in_proj_weight','k_proj_weight' )] = k_weight
                    sd[k.replace('in_proj_weight','v_proj_weight' )] = v_weight
                    del sd[k]
                
        missing, unexpected = self.load_state_dict(sd, strict=False)
        rank_zero_info(f"[TETFAllTalkingA2VqvaeKeylaPoseModel] Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        missing = [k for k in missing if not ("text" in k) or ('zero' in k)]
        if len(missing) > 0:
            rank_zero_info(f"[TETFAllTalkingA2VqvaeKeylaPoseModel] Missing Keys: {missing}")
            rank_zero_info(f"[TETFAllTalkingA2VqvaeKeylaPoseModel] Unexpected Keys: {unexpected}")    
            
    def configure_optimizers(self):
        lr = self.learning_rate
        if self.only_update_text and not self.update_text_and_q:
            rank_zero_info(f"[TETFAllTalkingA2VqvaeKeylaPoseModel] Only update text params")
            self.requires_grad_(False)
            text_paras = []
            for name, param in self.named_parameters():
                if 'instruct_encoder' in name or ('text' in name and 'clip_text' not in name) or ("zero" in name) or ('au_classifier' in name):
                    text_paras.append(param)
                    param.requires_grad = True
                        
            to_update_paras = text_paras
            
        elif self.update_text_and_q:
            rank_zero_info(f"[TETFAllTalkingA2VqvaeKeylaPoseModel] Only update text and query")
            self.requires_grad_(False)
            text_and_q_paras = []
            for name, param in self.named_parameters():
                if 'instruct_encoder' in name or ('text' in name and 'clip_text' not in name) or ('backbone.conformer_layers' in name and 'q_proj_weight' in name) or ("zero" in name) or ('au_classifier' in name):
                    # print(name)
                    text_and_q_paras.append(param)
                    param.requires_grad = True
                
                        
            to_update_paras = text_and_q_paras
        else:
            to_update_paras = self.parameters()
        if self.zero_optim:
            opt = ZeroRedundancyOptimizer(to_update_paras,
                                          optimizer_class=torch.optim.AdamW,
                                          lr=lr,
                                          betas=(0.9, 0.98),
                                          weight_decay=self.weight_decay)
        else:
            opt = torch.optim.AdamW(to_update_paras,
                                    lr=lr,
                                    betas=(0.9, 0.98),
                                    weight_decay=self.weight_decay)
        # https://lightning.ai/docs/pytorch/stable/common/optimization.html#bring-your-own-custom-learning-rate-schedulers
        if self.use_scheduler:
            rank_zero_info(f"[TETFAllTalkingA2VqvaeKeylaPoseModel] Use lr scheduler: {self.lr_scheduler_config.target}")
            if 'inverse_sqrt' in self.lr_scheduler_config.target:
                scheduler = transformers.get_inverse_sqrt_schedule(optimizer=opt, num_warmup_steps=self.lr_scheduler_config.params.num_warmup_steps)
                # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
                lr_scheduler = {
                    'scheduler': scheduler,
                    'name': 'main-LR',
                    'interval': 'step',
                    'frequency': self.lr_scheduler_config.params.frequency if hasattr(self.lr_scheduler_config.params, 'frequency') else 1
                    }
            else:
                lr_scheduler = instantiate_lrscheduler_from_config(opt, self.lr_scheduler_config)
            return [opt], [lr_scheduler]
        return opt
    
    def get_pose_loss(self, pred, target, masks, audio_flag):
         
        if self.loss_scale_by_dim:
            masks = masks.to(dtype=pred.dtype).repeat(1, 1, pred.size(-1))
        else:
            masks = masks.to(dtype=pred.dtype)
        mse_loss = F.mse_loss(pred, target, reduction='none')
        audio_flag = repeat(audio_flag, 'b -> b t 3', t=pred.shape[1])
        mse_loss = mse_loss * audio_flag
        masks = self._weights_nonzero(target, mse_loss.dtype)
        if masks.sum() == 0:
            return (mse_loss * masks).sum()
        else:
            loss = (mse_loss * masks).sum() / masks.sum()
            return loss
        
    def get_au_class_loss(self, x0_pred, au_sentence, masks):
        mean_x0_pred = torch.sum(x0_pred * masks, dim=1) / torch.sum(masks, dim=1)
        pred_logits = self.au_classifier(mean_x0_pred)
        label_tensor_list = map(get_au_label, au_sentence)
        label_tensor = torch.stack(list(label_tensor_list), dim=0).to(pred_logits.device)
        loss = nn.BCEWithLogitsLoss()(pred_logits, label_tensor)
        return loss
        
    def get_input(self, batch):
        latent = batch['latent'].to(memory_format=torch.contiguous_format)  # some padding 0s are at the end
        audio_feat = batch['audio_feat'].to(memory_format=torch.contiguous_format)
        keylatent = batch['keylatent'].to(memory_format=torch.contiguous_format)
        pose = batch['headpose'].to(memory_format=torch.contiguous_format)
        if not self.inst_use_parapharse:
            instruct_sen = batch['instructs']
        else:
            instruct_sen = list(map(self.all_paraphrase, batch['instructs'], batch['inst_flag'], batch['audio_flag']))
        
        if self.use_au:
            instruct_sen = list(map(self.add_au_func, instruct_sen, batch['inst_flag'], batch['audio_flag'], batch['action_units']))
        with torch.no_grad():
            instruct_feat = self.clip_text_encoder.encode(instruct_sen).to(memory_format=torch.contiguous_format)
        
        # get mask
        masks = make_non_pad_mask(batch['latent_length'], xs=batch['latent'][..., 0])
        masks = masks.to(memory_format=torch.contiguous_format)  #  (bs, max_num_frames)
        masks = rearrange(masks, 'b t -> b t 1').contiguous()
        # masks: (bs, T, 1)
        return audio_feat, keylatent, latent, masks, instruct_feat, pose

    def get_learned_conditioning(self, audio_feat, keylatent, masks, instructions, pose=None):
        # audio_feat: (bs, T*2, 512)
        # keylatent: (bs, T, 512)
        # get downsampled audio feat
        
        audio_feat = self.audio_downsample(audio_feat.transpose(1, 2)).transpose(1, 2)
        # audio_feat: (bs, T, 512)
        # audio encoder
        audio_feat = self.audio_encoder(audio_feat, masks=masks)
        # get keyframe latent
        keylatent_feat = self.keylatent_encoder(keylatent)
        # keylatent_feat: (bs, T, 512)
        
        inst_feat = self.instruct_encoder(instructions)
        out_pose = self.pose_pred(audio_feat, ~masks)  # (bs, T, 3)
        # out_pose = torch.rand_like(pose, device=pose.device)
        if pose is not None:
            audio_feat += self.pose_proj(pose)
        else:
            audio_feat += self.pose_proj(out_pose)  # (bs, T, d)
        
        return audio_feat, keylatent_feat, inst_feat, out_pose

   

    def forward(self, latent, cond1, cond2, cond3, audio_flag, inst_flag, mu=0.0, masks=None):
        t = torch.rand(latent.shape[0], dtype=latent.dtype, device=latent.device,
                       requires_grad=False)
        t = torch.clamp(t, self.offset, 1.0 - self.offset)

        # get xt and noise
        xt, z = self.forward_diffusion(latent, t, mu)  # xt, z: (bs, T, 512)

        audio_flag = repeat(audio_flag, 'b -> b f d', f = cond1.shape[1], d = cond1.shape[-1])
        inst_flag = repeat(inst_flag, 'b -> b f d', f = cond3.shape[1], d = cond3.shape[-1])
        
        cond1 = cond1 * audio_flag
        cond3 = cond3 * inst_flag
        x0_estimation = self.backbone(input=xt,
                                      diffusion_step=t,
                                      cond1=cond1,
                                      cond2=cond2,
                                      cond3=cond3,
                                      masks=masks)
        # x0_estimation: (bs, T, 512)

        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        rho = x0_estimation * torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        lambda_ = 1.0 - torch.exp(-cum_noise)

        noise_pred = (xt - rho) / (torch.sqrt(lambda_))
        return x0_estimation, noise_pred, z



    def training_step(self, batch, batch_idx):
        """
        batch keys:
        latent: (bs, T, 512)
        audio_feat: (bs, T*2, 512)
        keylatent: (bs, T, 512)
        """
        audio_feat, keylatent, latent, masks, inst_feat, pose = self.get_input(batch)
        audio_feat, keylatent_feat, inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat,  pose=pose)
        # audio_feat: (bs, T, 512)
        # keylatent_feat: (bs, T, 512)
        x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks, inst_flag=batch['inst_flag'], audio_flag=batch['audio_flag'])

        # log to lightning
        loss_pose = self.get_pose_loss(pose_pred, pose, masks, batch['audio_flag'])
        loss_x0 = self.get_loss(x0_pred, latent, masks)
        loss_noise = self.get_loss(noise_pred, noise_gt, masks)
        loss_au_class = self.get_au_class_loss(x0_pred, batch['action_units'], masks) if self.use_au_class_loss else 0.0
        # loss_smooth = self.get_loss(x0_pred[:, 1:], x0_pred[:, :-1], masks[:, 1:]) if self.loss_smooth_weight > 0.0 else 0.0

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_pose': loss_pose.mean() * self.diffusion_loss_pose_weight})
        loss_dict.update({f'{log_prefix}/loss_x0': loss_x0.mean()})
        loss_dict.update({f'{log_prefix}/loss_noise': loss_noise})
        loss_dict.update({f'{log_prefix}/loss_au_class': loss_au_class * self.au_class_loss_weight})
        if self.use_au_class_loss:
            loss = self.diffusion_loss_pose_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise + self.au_class_loss_weight * loss_au_class
        else:
            loss = self.diffusion_loss_pose_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise 
        loss_dict.update({f'{log_prefix}/loss': loss})

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            # self.log(f'{log_prefix}/lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # log to tensorboard
        # https://www.pytorchlightning.ai/blog/tensorboard-with-pytorch-lightning
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_pose', loss_pose.mean(), self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_x0', loss_x0.mean(), self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_noise', loss_noise, self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_au_class', loss_au_class, self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss', loss, self.global_step)
        # self.logger.experiment.add_scalar(f'{log_prefix}/loss_smooth', loss_smooth, self.global_step)
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        audio_feat, keylatent,latent, masks, inst_feat, pose = self.get_input(batch)
        audio_feat, keylatent_feat, inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat, pose=pose)
        # audio_feat: (bs, T, 512)
        # keylatent_feat: (bs, T, 512)
        x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks, inst_flag=batch['inst_flag'], audio_flag=batch['audio_flag'])

        loss_pose = self.get_pose_loss(pose_pred, pose, masks, batch['audio_flag'])
        loss_x0 = self.get_loss(x0_pred, latent, masks)
        loss_noise = self.get_loss(noise_pred, noise_gt, masks)
        loss_au_class = self.get_au_class_loss(x0_pred, batch['action_units'], masks) if self.use_au_class_loss else 0.0

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_pose': loss_pose.mean() * self.diffusion_loss_pose_weight})
        loss_dict.update({f'{log_prefix}/loss_x0': loss_x0.mean() * self.diffusion_loss_x0_weight})
        loss_dict.update({f'{log_prefix}/loss_noise': loss_noise * self.diffusion_loss_noise_weight})
        loss_dict.update({f'{log_prefix}/loss_au_class': loss_au_class * self.au_class_loss_weight})

        if self.use_au_class_loss:
            loss = self.diffusion_loss_pose_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise + self.au_class_loss_weight * loss_au_class
        else:
            loss = self.diffusion_loss_pose_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise 
        loss_dict.update({f'{log_prefix}/loss': loss})

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        
    @torch.no_grad()
    def reverse_diffusion(self, z, cond1, cond2, cond3, n_timesteps, audio_flag, inst_flag, mu=0.0, masks=None):
        h = 1.0 / n_timesteps
        xt = z
        audio_flag = repeat(audio_flag, 'b -> b f d', f = cond1.shape[1], d = cond1.shape[-1])
        inst_flag = repeat(inst_flag, 'b -> b f d', f = cond3.shape[1], d = cond3.shape[-1])
        
        cond1 = cond1 * audio_flag
        cond3 = cond3 * inst_flag
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = self._get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            # xt: (bs, T, 512)
            # t: (bs)
            # cond: (bs, T, 512)
            x0_ = self.backbone(input=xt,
                                diffusion_step=t,
                                cond1=cond1,
                                cond2=cond2,
                                cond3=cond3,
                                masks=masks)

            cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
            rho = x0_*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
            noise_pred = xt - rho
            lambda_ = 1.0 - torch.exp(-cum_noise)
            logp = - noise_pred / (lambda_ + 1e-8)
            dxt = 0.5 * (mu - xt - logp)
            dxt = dxt * noise_t * h
            xt = xt - dxt
        return xt

    @torch.no_grad()
    def sample(self, z_shape, cond1, cond2,cond3, audio_flag, inst_flag, n_timesteps=150, temperature=1.8, mu=0.0, masks=None):
        # z_shape: latent.shape
        # *_flag: (bs)
        z = mu + torch.randn(z_shape, dtype=cond1.dtype, device=cond1.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.reverse_diffusion(z=z, cond1=cond1, cond2=cond2, cond3=cond3, n_timesteps=n_timesteps, mu=mu, masks=masks, audio_flag=audio_flag, inst_flag=inst_flag)
        return decoder_outputs


    @torch.no_grad()
    def evaluate(self, audio_feat, keylatent, diff_init_shape, inst_feat, pose=None, audio_flag=True, inst_flag=True):
        """evaluate a2e from the given audio_feat and keylatent
        audio_feat: (bs, T*2, 512)
        keylatent: (bs, T, 768)
        """
        # get keyframe latent (bs, T, 768)
        keylatent = keylatent.repeat(1, audio_feat.shape[1]//2, 1)

        # get mask
        latent_length = torch.tensor([audio_feat.shape[1]//2]).to(audio_feat.device)
        masks = make_non_pad_mask(latent_length, xs=keylatent[..., 0]).to(memory_format=torch.contiguous_format)
        masks = rearrange(masks, 'b t -> b t 1').contiguous()
        
        audio_feat, keylatent_feat, inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat, pose=pose)
        if isinstance(diff_init_shape, int):
            diff_init_shape = (audio_feat.shape[0], audio_feat.shape[1], diff_init_shape)
        audio_flag_vec = torch.tensor([audio_flag], dtype=audio_feat.dtype).to(audio_feat.device)
        inst_flag_vec = torch.tensor([inst_flag], dtype=inst_feat.dtype).to(inst_feat.device)
        latent_rec = self.sample(diff_init_shape, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks, audio_flag=audio_flag_vec, inst_flag=inst_flag_vec)
        latent_rec = self._denormalize_latent(latent_rec)
        return latent_rec

    
    @torch.no_grad()
    def log_latents(self, batch, **kwargs):
        log = dict()
        audio_feat, keylatent, latent, masks, inst_feat, pose = self.get_input(batch)
        audio_feat, keylatent_feat,  inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat)
        # if self.val_with_extra_pose_npy_path is not None:
        #     # '/data/datasets/HDTF/clean_videos_split10s_ptcode/head_pose/RD_Radio10_000-00000.npy'
        #     # '/data/datasets/HDTF/clean_videos_split10s_ptcode/head_pose/RD_Radio11_000-00000.npy'
        #     rank_zero_info(f"[TalkingA2VqvaeKeylaPoseModel] Use extra pose: {self.val_with_extra_pose_npy_path}")
        #     extra_pose = np.load(self.val_with_extra_pose_npy_path)
        #     extra_pose = torch.from_numpy(extra_pose).float().to(pose.device)[:pose.shape[1], :].unsqueeze(0)
        #     audio_feat, keylatent_feat, out_pose = self.get_learned_conditioning(audio_feat, keylatent, masks, pose=extra_pose)
        # audio_feat: (bs, T, 512)
        # keylatent_feat: (bs, T, 512)
        latent_rec = self.sample(latent.shape, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, audio_flag=batch['audio_flag'], inst_flag=batch['inst_flag'], masks=masks)
        log["latent"] = self._denormalize_latent(latent)
        log["latent_rec"] = self._denormalize_latent(latent_rec)
        log["latent_filename"] = batch['latent_filename']
        return log

class TETFAllTextShortCutTalkingA2VqvaeKeylaPoseModel(TETFAllTalkingA2VqvaeKeylaPoseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, latent, cond1, cond2, cond3, audio_flag, inst_flag, mu=0.0, masks=None):
        t = torch.rand(latent.shape[0], dtype=latent.dtype, device=latent.device,
                       requires_grad=False)
        t = torch.clamp(t, self.offset, 1.0 - self.offset)

        # get xt and noise
        xt, z = self.forward_diffusion(latent, t, mu)  # xt, z: (bs, T, 512)

        audio_flag = repeat(audio_flag, 'b -> b f d', f = cond1.shape[1], d = cond1.shape[-1])
        # inst_flag = repeat(inst_flag, 'b -> b f d', f = cond3.shape[1], d = cond3.shape[-1])
        
        cond1 = cond1 * audio_flag
        # cond3 = cond3 * inst_flag
        x0_estimation = self.backbone(input=xt,
                                      diffusion_step=t,
                                      cond1=cond1,
                                      cond2=cond2,
                                      cond3=cond3,
                                      inst_flag=inst_flag,
                                      masks=masks)
        # x0_estimation: (bs, T, 512)

        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        rho = x0_estimation * torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        lambda_ = 1.0 - torch.exp(-cum_noise)

        noise_pred = (xt - rho) / (torch.sqrt(lambda_))
        return x0_estimation, noise_pred, z
    
    @torch.no_grad()
    def reverse_diffusion(self, z, cond1, cond2, cond3, n_timesteps, audio_flag, inst_flag, mu=0.0, masks=None):
        h = 1.0 / n_timesteps
        xt = z
        audio_flag = repeat(audio_flag, 'b -> b f d', f = cond1.shape[1], d = cond1.shape[-1])
        # inst_flag = repeat(inst_flag, 'b -> b f d', f = cond3.shape[1], d = cond3.shape[-1])
        
        cond1 = cond1 * audio_flag
        # cond3 = cond3 * inst_flag
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = self._get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            # xt: (bs, T, 512)
            # t: (bs)
            # cond: (bs, T, 512)
            x0_ = self.backbone(input=xt,
                                diffusion_step=t,
                                cond1=cond1,
                                cond2=cond2,
                                cond3=cond3,
                                inst_flag=inst_flag,
                                masks=masks)

            cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
            rho = x0_*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
            noise_pred = xt - rho
            lambda_ = 1.0 - torch.exp(-cum_noise)
            logp = - noise_pred / (lambda_ + 1e-8)
            dxt = 0.5 * (mu - xt - logp)
            dxt = dxt * noise_t * h
            xt = xt - dxt
        return xt
    
    
class TETFAllSpaceAlignTalkingA2VqvaeKeylaPoseModel(TETFAllTalkingA2VqvaeKeylaPoseModel):
    def __init__(self, use_text_adapter=True, repeat_style_token=5, text_adapter_beta=0.6, first_stage=False, space_cosine_loss_weight=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_text_adapter = use_text_adapter
        if self.use_text_adapter:
            self.text_adapter = Adapter(self.model_dim)
            self.text_adapter_beta = text_adapter_beta
        self.video_style_encoder = StyleEncoder(d_model=768, input_dim=self.model_dim, pos_embed_len=250, aggregate_method="self_attention_pooling")
        self.first_stage = first_stage
        self.space_cosine_loss_weight = space_cosine_loss_weight
        self.repeat_style_token = repeat_style_token
    def get_input(self, batch):
        latent = batch['latent'].to(memory_format=torch.contiguous_format)  # some padding 0s are at the end
        audio_feat = batch['audio_feat'].to(memory_format=torch.contiguous_format)
        keylatent = batch['keylatent'].to(memory_format=torch.contiguous_format)
        pose = batch['headpose'].to(memory_format=torch.contiguous_format)
        if not self.inst_use_parapharse:
            instruct_sen = batch['instructs']
        else:
            instruct_sen = list(map(self.all_paraphrase, batch['instructs'], batch['inst_flag'], batch['audio_flag']))
        
        if self.use_au:
            instruct_sen = list(map(self.add_au_func, instruct_sen, batch['inst_flag'], batch['audio_flag'], batch['action_units']))
        with torch.no_grad():
            instruct_feat = self.clip_text_encoder.encode_text(instruct_sen).to(memory_format=torch.contiguous_format) # only extract [EOS] token
        
        # get mask
        masks = make_non_pad_mask(batch['latent_length'], xs=batch['latent'][..., 0])
        masks = masks.to(memory_format=torch.contiguous_format)  #  (bs, max_num_frames)
        masks = rearrange(masks, 'b t -> b t 1').contiguous()
        # masks: (bs, T, 1)
        return audio_feat, keylatent, latent, masks, instruct_feat, pose
    
    def get_learned_conditioning(self, audio_feat, keylatent, masks, instructions, pose=None):
        # audio_feat: (bs, T*2, 512)
        # keylatent: (bs, T, 512)
        # get downsampled audio feat
        
        audio_feat = self.audio_downsample(audio_feat.transpose(1, 2)).transpose(1, 2)
        # audio_feat: (bs, T, 512)
        # audio encoder
        audio_feat = self.audio_encoder(audio_feat, masks=masks)
        # get keyframe latent
        keylatent_feat = self.keylatent_encoder(keylatent)
        # keylatent_feat: (bs, T, 512)
        
        inst_feat = self.instruct_encoder(instructions)
        if self.use_text_adapter:
            inst_feat_adapter = self.text_adapter(inst_feat)
        inst_feat =(1 - self.text_adapter_beta) * inst_feat + self.text_adapter_beta * inst_feat_adapter
        inst_feat = inst_feat.unsqueeze(1).repeat(1, self.repeat_style_token, 1) # only use [CLS] token
        out_pose = self.pose_pred(audio_feat, ~masks)  # (bs, T, 3)
        if pose is not None:
            audio_feat += self.pose_proj(pose)
        else:
            audio_feat += self.pose_proj(out_pose)  # (bs, T, d)
        
        return audio_feat, keylatent_feat, inst_feat, out_pose    
    
    def training_step(self, batch, batch_idx):
        """
        batch keys:
        latent: (bs, T, 512)
        audio_feat: (bs, T*2, 512)
        keylatent: (bs, T, 512)
        """
        audio_feat, keylatent, latent, masks, inst_feat, pose = self.get_input(batch)
        audio_feat, keylatent_feat, inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat,  pose=pose)
        video_style = self.video_style_encoder(latent, pad_mask=masks.squeeze(-1)).unsqueeze(1).repeat(1, self.repeat_style_token, 1)
        if self.first_stage:
            x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, cond3=video_style, masks=masks, inst_flag=batch['inst_flag'], audio_flag=batch['audio_flag'])
        else:
            text_prob = random.random()
            if text_prob < 0.5:   
                x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks, inst_flag=batch['inst_flag'], audio_flag=batch['audio_flag'])
            else:
                x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, cond3=video_style, masks=masks, inst_flag=batch['inst_flag'], audio_flag=batch['audio_flag'])

        # log to lightning
        loss_pose = self.get_pose_loss(pose_pred, pose, masks, batch['audio_flag'])
        loss_x0 = self.get_loss(x0_pred, latent, masks)
        loss_noise = self.get_loss(noise_pred, noise_gt, masks)
        loss_au_class = self.get_au_class_loss(x0_pred, batch['action_units'], masks) if self.use_au_class_loss else 0.0
        # loss_smooth = self.get_loss(x0_pred[:, 1:], x0_pred[:, :-1], masks[:, 1:]) if self.loss_smooth_weight > 0.0 else 0.0
      
        loss_space_cosine = torch.mean((1 - torch.cosine_similarity(video_style[:,0,:], inst_feat[:,0,:], dim=-1))) if not self.first_stage else 0.0
        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_pose': loss_pose.mean() * self.diffusion_loss_pose_weight})
        loss_dict.update({f'{log_prefix}/loss_x0': loss_x0.mean()})
        loss_dict.update({f'{log_prefix}/loss_noise': loss_noise})
        loss_dict.update({f'{log_prefix}/loss_au_class': loss_au_class * self.au_class_loss_weight})
        loss_dict.update({f'{log_prefix}/loss_space_cosine': loss_space_cosine * self.space_cosine_loss_weight})

        if self.use_au_class_loss:
            loss = self.diffusion_loss_pose_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise + self.au_class_loss_weight * loss_au_class + self.space_cosine_loss_weight * loss_space_cosine
        else:
            loss = self.diffusion_loss_pose_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise + self.space_cosine_loss_weight * loss_space_cosine
        loss_dict.update({f'{log_prefix}/loss': loss})

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            # self.log(f'{log_prefix}/lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # log to tensorboard
        # https://www.pytorchlightning.ai/blog/tensorboard-with-pytorch-lightning
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_pose', loss_pose.mean(), self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_x0', loss_x0.mean(), self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_noise', loss_noise, self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_au_class', loss_au_class, self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_space_cosine', loss_space_cosine, self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss', loss, self.global_step)
        # self.logger.experiment.add_scalar(f'{log_prefix}/loss_smooth', loss_smooth, self.global_step)
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        audio_feat, keylatent,latent, masks, inst_feat, pose = self.get_input(batch)
        audio_feat, keylatent_feat, inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat, pose=pose)
        video_style = self.video_style_encoder(latent, pad_mask=masks.squeeze(-1)).unsqueeze(1).repeat(1, self.repeat_style_token, 1)
        # audio_feat: (bs, T, 512)
        # keylatent_feat: (bs, T, 512)
        if self.first_stage:
            x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, cond3=video_style, masks=masks, inst_flag=batch['inst_flag'], audio_flag=batch['audio_flag'])
        else:
            text_prob = random.random()
            if text_prob < 0.5:   
                x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks, inst_flag=batch['inst_flag'], audio_flag=batch['audio_flag'])
            else:
                x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, cond3=video_style, masks=masks, inst_flag=batch['inst_flag'], audio_flag=batch['audio_flag'])

        loss_pose = self.get_pose_loss(pose_pred, pose, masks, batch['audio_flag'])
        loss_x0 = self.get_loss(x0_pred, latent, masks)
        loss_noise = self.get_loss(noise_pred, noise_gt, masks)
        loss_au_class = self.get_au_class_loss(x0_pred, batch['action_units'], masks) if self.use_au_class_loss else 0.0
        loss_space_cosine = torch.mean((1 - torch.cosine_similarity(video_style[:,0,:], inst_feat[:,0,:], dim=-1))) if not self.first_stage else 0.0

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_pose': loss_pose.mean() * self.diffusion_loss_pose_weight})
        loss_dict.update({f'{log_prefix}/loss_x0': loss_x0.mean() * self.diffusion_loss_x0_weight})
        loss_dict.update({f'{log_prefix}/loss_noise': loss_noise * self.diffusion_loss_noise_weight})
        loss_dict.update({f'{log_prefix}/loss_au_class': loss_au_class * self.au_class_loss_weight})
        loss_dict.update({f'{log_prefix}/loss_space_cosine': loss_space_cosine * self.space_cosine_loss_weight})

        if self.use_au_class_loss:
            loss = self.diffusion_loss_pose_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise + self.au_class_loss_weight * loss_au_class + self.space_cosine_loss_weight * loss_space_cosine
        else:
            loss = self.diffusion_loss_pose_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise + self.space_cosine_loss_weight * loss_space_cosine
        loss_dict.update({f'{log_prefix}/loss': loss})

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        
class UnifiedTETFAllTalkingA2VqvaeKeylaPoseModel(TETFAllTalkingA2VqvaeKeylaPoseModel):
    def __init__(self, only_update_text=False, update_text_and_q=False, inst_use_parapharse=True, use_au=False, use_au_class_loss=False, au_class_loss_weight=0, minor_init_text_cross_attn=False, use_extend_emotion_para=False, use_intensity_emotion=False, use_para_au=False, *args, **kwargs):
        super().__init__(only_update_text, update_text_and_q, inst_use_parapharse, use_au, use_au_class_loss, au_class_loss_weight, minor_init_text_cross_attn, *args, **kwargs)
        self.use_extend_emotion_para = use_extend_emotion_para
        self.use_intensity_emotion = use_intensity_emotion
        self.use_para_au = use_para_au
        self.all_paraphrase_extend = all_paraphrase_extend
        self.all_paraphrase_intensity = all_paraphrase_intensity
       
            
    def get_input(self, batch):
        latent = batch['latent'].to(memory_format=torch.contiguous_format)  # some padding 0s are at the end
        audio_feat = batch['audio_feat'].to(memory_format=torch.contiguous_format)
        keylatent = batch['keylatent'].to(memory_format=torch.contiguous_format)
        pose = batch['headpose'].to(memory_format=torch.contiguous_format)
        if not self.inst_use_parapharse:
            instruct_sen = batch['instructs']
        elif self.use_extend_emotion_para and not self.use_intensity_emotion:
            instruct_sen = list(map(self.all_paraphrase_extend, batch['instructs'], batch['inst_flag'], batch['audio_flag']))
        elif self.use_intensity_emotion:
            # print('use intensity emotion')
            instruct_sen = list(map(self.all_paraphrase_intensity, batch['instructs'], batch['inst_flag'], batch['audio_flag'], batch['latent_filename']))
        else:
            instruct_sen = list(map(self.all_paraphrase, batch['instructs'], batch['inst_flag'], batch['audio_flag']))
        
        if self.use_au:
            instruct_sen = list(map(self.add_au_func, instruct_sen, batch['inst_flag'], batch['audio_flag'], batch['action_units']))
        with torch.no_grad():
            instruct_feat = self.clip_text_encoder.encode(instruct_sen).to(memory_format=torch.contiguous_format) 
        
        # get mask
        masks = make_non_pad_mask(batch['latent_length'], xs=batch['latent'][..., 0])
        masks = masks.to(memory_format=torch.contiguous_format)  #  (bs, max_num_frames)
        masks = rearrange(masks, 'b t -> b t 1').contiguous()
        # masks: (bs, T, 1)
        return audio_feat, keylatent, latent, masks, instruct_feat, pose    

    def get_learned_conditioning(self, audio_feat, keylatent, masks, instructions, audio_flag, pose=None):
        # audio_feat: (bs, T*2, 512)
        # keylatent: (bs, T, 512)
        # get downsampled audio feat
        
        audio_feat = self.audio_downsample(audio_feat.transpose(1, 2)).transpose(1, 2)
        # audio_feat: (bs, T, 512)
        # audio encoder
        audio_feat = self.audio_encoder(audio_feat, masks=masks)
        # get keyframe latent
        keylatent_feat = self.keylatent_encoder(keylatent)
        # keylatent_feat: (bs, T, 512)
        
        inst_feat = self.instruct_encoder(instructions)
        out_pose = self.pose_pred(audio_feat, ~masks)  # (bs, T, 3)
        # out_pose = torch.rand_like(pose, device=pose.device)
        if pose is not None:
            t2m_data = torch.nonzero(1 - audio_flag).squeeze(-1)
            pose = pose.to(out_pose.dtype)
            pose[t2m_data] = out_pose[t2m_data]
            audio_feat += self.pose_proj(pose)
        else:
            audio_feat += self.pose_proj(out_pose)  # (bs, T, d)
        
        return audio_feat, keylatent_feat, inst_feat, out_pose

   

    def forward(self, latent, cond1, cond2, cond3, audio_flag, inst_flag, mu=0.0, masks=None):
        t = torch.rand(latent.shape[0], dtype=latent.dtype, device=latent.device,
                       requires_grad=False)
        t = torch.clamp(t, self.offset, 1.0 - self.offset)

        # get xt and noise
        xt, z = self.forward_diffusion(latent, t, mu)  # xt, z: (bs, T, 512)

        # audio_flag = repeat(audio_flag, 'b -> b f d', f = cond1.shape[1], d = cond1.shape[-1])
        # inst_flag = repeat(inst_flag, 'b -> b f d', f = cond3.shape[1], d = cond3.shape[-1])
        
        # cond1 = cond1 * audio_flag
        # cond3 = cond3 * inst_flag
        x0_estimation = self.backbone(input=xt,
                                      diffusion_step=t,
                                      cond1=cond1,
                                      cond2=cond2,
                                      cond3=cond3,
                                      masks=masks)
        # x0_estimation: (bs, T, 512)

        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        rho = x0_estimation * torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        lambda_ = 1.0 - torch.exp(-cum_noise)

        noise_pred = (xt - rho) / (torch.sqrt(lambda_))
        return x0_estimation, noise_pred, z



    def training_step(self, batch, batch_idx):
        """
        batch keys:
        latent: (bs, T, 512)
        audio_feat: (bs, T*2, 512)
        keylatent: (bs, T, 512)
        """
        audio_feat, keylatent, latent, masks, inst_feat, pose = self.get_input(batch)
        audio_feat, keylatent_feat, inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat, batch['audio_flag'],  pose=pose)
        # audio_feat: (bs, T, 512)
        # keylatent_feat: (bs, T, 512)
        x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks, inst_flag=batch['inst_flag'], audio_flag=batch['audio_flag'])

        # log to lightning
        loss_pose = self.get_pose_loss(pose_pred, pose, masks, batch['audio_flag'])
        loss_x0 = self.get_loss(x0_pred, latent, masks)
        loss_noise = self.get_loss(noise_pred, noise_gt, masks)
        loss_au_class = self.get_au_class_loss(x0_pred, batch['action_units'], masks) if self.use_au_class_loss else 0.0
        # loss_smooth = self.get_loss(x0_pred[:, 1:], x0_pred[:, :-1], masks[:, 1:]) if self.loss_smooth_weight > 0.0 else 0.0

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_pose': loss_pose.mean() * self.diffusion_loss_pose_weight})
        loss_dict.update({f'{log_prefix}/loss_x0': loss_x0.mean()})
        loss_dict.update({f'{log_prefix}/loss_noise': loss_noise})
        loss_dict.update({f'{log_prefix}/loss_au_class': loss_au_class * self.au_class_loss_weight})
        if self.use_au_class_loss:
            loss = self.diffusion_loss_pose_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise + self.au_class_loss_weight * loss_au_class
        else:
            loss = self.diffusion_loss_pose_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise 
        loss_dict.update({f'{log_prefix}/loss': loss})

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            # self.log(f'{log_prefix}/lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # log to tensorboard
        # https://www.pytorchlightning.ai/blog/tensorboard-with-pytorch-lightning
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_pose', loss_pose.mean(), self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_x0', loss_x0.mean(), self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_noise', loss_noise, self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss_au_class', loss_au_class, self.global_step)
        self.logger.experiment.add_scalar(f'{log_prefix}/loss', loss, self.global_step)
        # self.logger.experiment.add_scalar(f'{log_prefix}/loss_smooth', loss_smooth, self.global_step)
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        audio_feat, keylatent,latent, masks, inst_feat, pose = self.get_input(batch)
        audio_feat, keylatent_feat, inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat,batch['audio_flag'], pose=pose)
        # audio_feat: (bs, T, 512)
        # keylatent_feat: (bs, T, 512)
        x0_pred, noise_pred, noise_gt = self(latent, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks, inst_flag=batch['inst_flag'], audio_flag=batch['audio_flag'])

        loss_pose = self.get_pose_loss(pose_pred, pose, masks, batch['audio_flag'])
        loss_x0 = self.get_loss(x0_pred, latent, masks)
        loss_noise = self.get_loss(noise_pred, noise_gt, masks)
        loss_au_class = self.get_au_class_loss(x0_pred, batch['action_units'], masks) if self.use_au_class_loss else 0.0

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_pose': loss_pose.mean() * self.diffusion_loss_pose_weight})
        loss_dict.update({f'{log_prefix}/loss_x0': loss_x0.mean() * self.diffusion_loss_x0_weight})
        loss_dict.update({f'{log_prefix}/loss_noise': loss_noise * self.diffusion_loss_noise_weight})
        loss_dict.update({f'{log_prefix}/loss_au_class': loss_au_class * self.au_class_loss_weight})

        if self.use_au_class_loss:
            loss = self.diffusion_loss_pose_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise + self.au_class_loss_weight * loss_au_class
        else:
            loss = self.diffusion_loss_pose_weight*loss_pose + self.diffusion_loss_x0_weight * loss_x0 + self.diffusion_loss_noise_weight * loss_noise 
        loss_dict.update({f'{log_prefix}/loss': loss})

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        
    @torch.no_grad()
    def reverse_diffusion(self, z, cond1, cond2, cond3, n_timesteps, audio_flag, inst_flag, mu=0.0, masks=None):
        h = 1.0 / n_timesteps
        xt = z
        # audio_flag = repeat(audio_flag, 'b -> b f d', f = cond1.shape[1], d = cond1.shape[-1])
        # inst_flag = repeat(inst_flag, 'b -> b f d', f = cond3.shape[1], d = cond3.shape[-1])
        
        # cond1 = cond1 * audio_flag
        # cond3 = cond3 * inst_flag
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = self._get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            # xt: (bs, T, 512)
            # t: (bs)
            # cond: (bs, T, 512)
            x0_ = self.backbone(input=xt,
                                diffusion_step=t,
                                cond1=cond1,
                                cond2=cond2,
                                cond3=cond3,
                                masks=masks)

            cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
            rho = x0_*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
            noise_pred = xt - rho
            lambda_ = 1.0 - torch.exp(-cum_noise)
            logp = - noise_pred / (lambda_ + 1e-8)
            dxt = 0.5 * (mu - xt - logp)
            dxt = dxt * noise_t * h
            xt = xt - dxt
        return xt

    @torch.no_grad()
    def sample(self, z_shape, cond1, cond2,cond3, audio_flag, inst_flag, n_timesteps=150, temperature=1.8, mu=0.0, masks=None):
        # z_shape: latent.shape
        # *_flag: (bs)
        z = mu + torch.randn(z_shape, dtype=cond1.dtype, device=cond1.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.reverse_diffusion(z=z, cond1=cond1, cond2=cond2, cond3=cond3, n_timesteps=n_timesteps, mu=mu, masks=masks, audio_flag=audio_flag, inst_flag=inst_flag)
        return decoder_outputs


    @torch.no_grad()
    def evaluate(self, audio_feat, keylatent, diff_init_shape, inst_feat, pose=None, audio_flag=True, inst_flag=True):
        """evaluate a2e from the given audio_feat and keylatent
        audio_feat: (bs, T*2, 512)
        keylatent: (bs, T, 768)
        """
        # get keyframe latent (bs, T, 768)
        keylatent = keylatent.repeat(1, audio_feat.shape[1]//2, 1)

        # get mask
        latent_length = torch.tensor([audio_feat.shape[1]//2]).to(audio_feat.device)
        masks = make_non_pad_mask(latent_length, xs=keylatent[..., 0]).to(memory_format=torch.contiguous_format)
        masks = rearrange(masks, 'b t -> b t 1').contiguous()
        
        audio_feat, keylatent_feat, inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat, None, pose=pose)
        if isinstance(diff_init_shape, int):
            diff_init_shape = (audio_feat.shape[0], audio_feat.shape[1], diff_init_shape)
        audio_flag_vec = torch.tensor([[audio_flag]], dtype=audio_feat.dtype).to(audio_feat.device)
        inst_flag_vec = torch.tensor([[inst_flag]], dtype=inst_feat.dtype).to(inst_feat.device)
        latent_rec = self.sample(diff_init_shape, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks, audio_flag=audio_flag_vec, inst_flag=inst_flag_vec)
        latent_rec = self._denormalize_latent(latent_rec)
        return latent_rec

    
    @torch.no_grad()
    def log_latents(self, batch, **kwargs):
        log = dict()
        audio_feat, keylatent, latent, masks, inst_feat, pose = self.get_input(batch)
        audio_feat, keylatent_feat,  inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat, batch['audio_flag'])
    
        latent_rec = self.sample(latent.shape, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, audio_flag=batch['audio_flag'], inst_flag=batch['inst_flag'], masks=masks)
        log["latent"] = self._denormalize_latent(latent)
        log["latent_rec"] = self._denormalize_latent(latent_rec)
        log["latent_filename"] = batch['latent_filename']
        return log
    
        
class UnifiedSplitTETFAllTalkingA2VqvaeKeylaPoseModel(UnifiedTETFAllTalkingA2VqvaeKeylaPoseModel):
    def  __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emotion_adapter = Adapter(self.middle_dim)
        self.emotion_adapter_beta = 0.6
        self.motion_adapter = Adapter(self.middle_dim)
        self.motion_adapter_beta = 0.6
        
        
    @torch.no_grad()
    def evaluate(self, audio_feat, keylatent, diff_init_shape, inst_feat, pose=None, audio_flag=True, inst_flag=True):
        """evaluate a2e from the given audio_feat and keylatent
        audio_feat: (bs, T*2, 512)
        keylatent: (bs, T, 768)
        """
        # get keyframe latent (bs, T, 768)
        keylatent = keylatent.repeat(1, audio_feat.shape[1]//2, 1)

        # get mask
        latent_length = torch.tensor([audio_feat.shape[1]//2]).to(audio_feat.device)
        masks = make_non_pad_mask(latent_length, xs=keylatent[..., 0]).to(memory_format=torch.contiguous_format)
        masks = rearrange(masks, 'b t -> b t 1').contiguous()
        audio_flag_vec = torch.tensor([audio_flag], dtype=audio_feat.dtype).to(audio_feat.device)
        inst_flag_vec = torch.tensor([inst_flag], dtype=inst_feat.dtype).to(inst_feat.device)
        audio_feat, keylatent_feat, inst_feat, pose_pred = self.get_learned_conditioning(audio_feat, keylatent, masks, inst_feat, audio_flag_vec, pose=pose)
        if isinstance(diff_init_shape, int):
            diff_init_shape = (audio_feat.shape[0], audio_feat.shape[1], diff_init_shape)

        latent_rec = self.sample(diff_init_shape, cond1=audio_feat, cond2=keylatent_feat, cond3=inst_feat, masks=masks, audio_flag=audio_flag_vec, inst_flag=inst_flag_vec)
        latent_rec = self._denormalize_latent(latent_rec)
        return latent_rec
    
        
    def get_learned_conditioning(self, audio_feat, keylatent, masks, instructions, audio_flag, pose=None):
        # audio_feat: (bs, T*2, 512)
        # keylatent: (bs, T, 512)
        # get downsampled audio feat
        
        audio_feat = self.audio_downsample(audio_feat.transpose(1, 2)).transpose(1, 2)
        # audio_feat: (bs, T, 512)
        # audio encoder
        audio_feat = self.audio_encoder(audio_feat, masks=masks)
        # get keyframe latent
        keylatent_feat = self.keylatent_encoder(keylatent)
        # keylatent_feat: (bs, T, 512)
        t2m_data = torch.nonzero(1 - audio_flag).squeeze(-1)
        other_data = torch.nonzero(audio_flag).squeeze(-1)
        inst_feat = self.instruct_encoder(instructions)
        
        # two branch adapter
        emotion_inst_feat = inst_feat[other_data]
        emotion_inst_feat = self.emotion_adapter(emotion_inst_feat)
        emotion_inst_feat = (1 - self.emotion_adapter_beta) * inst_feat[other_data] + self.emotion_adapter_beta * emotion_inst_feat
        motion_inst_feat = inst_feat[t2m_data]
        motion_inst_feat = self.motion_adapter(motion_inst_feat)
        motion_inst_feat = (1 - self.motion_adapter_beta) * inst_feat[t2m_data] + self.motion_adapter_beta * motion_inst_feat
        inst_feat[other_data] = emotion_inst_feat
        inst_feat[t2m_data] = motion_inst_feat
        
        out_pose = self.pose_pred(audio_feat, ~masks)  # (bs, T, 3)
        # out_pose = torch.rand_like(pose, device=pose.device)
        if pose is not None:
            t2m_data = torch.nonzero(1 - audio_flag).squeeze(-1)
            pose = pose.to(out_pose.dtype)
            pose[t2m_data] = out_pose[t2m_data]
            audio_feat += self.pose_proj(pose)
        else:
            audio_feat += self.pose_proj(out_pose)  # (bs, T, d)
        
        return audio_feat, keylatent_feat, inst_feat, out_pose
    
    def forward(self, latent, cond1, cond2, cond3, audio_flag, inst_flag, mu=0.0, masks=None):
        t = torch.rand(latent.shape[0], dtype=latent.dtype, device=latent.device,
                       requires_grad=False)
        t = torch.clamp(t, self.offset, 1.0 - self.offset)

        # get xt and noise
        xt, z = self.forward_diffusion(latent, t, mu)  # xt, z: (bs, T, 512)

        # audio_flag = repeat(audio_flag, 'b -> b f d', f = cond1.shape[1], d = cond1.shape[-1])
        # inst_flag = repeat(inst_flag, 'b -> b f d', f = cond3.shape[1], d = cond3.shape[-1])
        
        # cond1 = cond1 * audio_flag
        # cond3 = cond3 * inst_flag
        x0_estimation = self.backbone(input=xt,
                                      diffusion_step=t,
                                      cond1=cond1,
                                      cond2=cond2,
                                      cond3=cond3,
                                      audio_flag=audio_flag,
                                      inst_flag=inst_flag,
                                      masks=masks)
        # x0_estimation: (bs, T, 512)

        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        rho = x0_estimation * torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        lambda_ = 1.0 - torch.exp(-cum_noise)

        noise_pred = (xt - rho) / (torch.sqrt(lambda_))
        return x0_estimation, noise_pred, z
    
    @torch.no_grad()
    def reverse_diffusion(self, z, cond1, cond2, cond3, n_timesteps, audio_flag, inst_flag, mu=0.0, masks=None):
        h = 1.0 / n_timesteps
        xt = z
        # audio_flag = repeat(audio_flag, 'b -> b f d', f = cond1.shape[1], d = cond1.shape[-1])
        # inst_flag = repeat(inst_flag, 'b -> b f d', f = cond3.shape[1], d = cond3.shape[-1])
        
        # cond1 = cond1 * audio_flag
        # cond3 = cond3 * inst_flag
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = self._get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            # xt: (bs, T, 512)
            # t: (bs)
            # cond: (bs, T, 512)
            x0_ = self.backbone(input=xt,
                                diffusion_step=t,
                                cond1=cond1,
                                cond2=cond2,
                                cond3=cond3,
                                audio_flag=audio_flag,
                                inst_flag=inst_flag,
                                masks=masks)

            cum_noise = self._get_noise(time, self.beta_min, self.beta_max, cumulative=True)
            rho = x0_*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
            noise_pred = xt - rho
            lambda_ = 1.0 - torch.exp(-cum_noise)
            logp = - noise_pred / (lambda_ + 1e-8)
            dxt = 0.5 * (mu - xt - logp)
            dxt = dxt * noise_t * h
            xt = xt - dxt
        return xt


from multiprocessing.dummy import Pool as ThreadPool 

class UnifiedSplitEOSTETFAllTalkingA2VqvaeKeylaPoseModel(UnifiedSplitTETFAllTalkingA2VqvaeKeylaPoseModel):
    def  __init__(self, parallel_paraphrase=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel_paraphrase = parallel_paraphrase
        
    def get_input(self, batch):
        latent = batch['latent'].to(memory_format=torch.contiguous_format)  # some padding 0s are at the end
        audio_feat = batch['audio_feat'].to(memory_format=torch.contiguous_format)
        keylatent = batch['keylatent'].to(memory_format=torch.contiguous_format)
        pose = batch['headpose'].to(memory_format=torch.contiguous_format)
        if not self.inst_use_parapharse:
            instruct_sen = batch['instructs']
        elif self.use_extend_emotion_para and not self.use_intensity_emotion:
            if self.parallel_paraphrase:
                pool = ThreadPool(32)
                instruct_sen = pool.starmap(self.all_paraphrase_extend, zip(batch['instructs'], batch['inst_flag'], batch['audio_flag']))
                pool.close()
                pool.join()
            else:
                instruct_sen = list(map(self.all_paraphrase_extend, batch['instructs'], batch['inst_flag'], batch['audio_flag']))
        elif self.use_intensity_emotion:
            if self.parallel_paraphrase:
                pool = ThreadPool(32)
                instruct_sen = pool.starmap(self.all_paraphrase_intensity, zip(batch['instructs'], batch['inst_flag'], batch['audio_flag'], batch['latent_filename']))
                pool.close()
                pool.join()
            else:
                instruct_sen = list(map(self.all_paraphrase_intensity, batch['instructs'], batch['inst_flag'], batch['audio_flag'], batch['latent_filename']))
        else:
            if self.parallel_paraphrase:
                pool = ThreadPool(32)
                instruct_sen = pool.starmap(self.all_paraphrase, zip(batch['instructs'], batch['inst_flag'], batch['audio_flag']))
                pool.close()
                pool.join()
            else:
                instruct_sen = list(map(self.all_paraphrase, batch['instructs'], batch['inst_flag'], batch['audio_flag']))
        
        if self.use_au and not self.use_para_au:
            if self.parallel_paraphrase:
                pool = ThreadPool(32)
                instruct_sen = pool.starmap(self.add_au_func, zip(instruct_sen, batch['inst_flag'], batch['audio_flag'], batch['action_units']))
                pool.close()
                pool.join()
            else:
                instruct_sen = list(map(self.add_au_func, instruct_sen, batch['inst_flag'], batch['audio_flag'], batch['action_units']))
        elif self.use_para_au:
            if self.parallel_paraphrase:
                pool = ThreadPool(32)
                instruct_sen = pool.starmap(self.add_au_func_para, zip(instruct_sen, batch['inst_flag'], batch['audio_flag'], batch['action_units'], batch['latent_filename']))
                pool.close()
                pool.join()
            else:
                instruct_sen = list(map(self.add_au_func_para, instruct_sen, batch['inst_flag'], batch['audio_flag'], batch['action_units'], batch['latent_filename']))
        with torch.no_grad():
            motion_idx = torch.nonzero(1 - batch['audio_flag']).squeeze(-1)
            emotion_idx = torch.nonzero(batch['audio_flag']).squeeze(-1)
            motion_instruct_sen = [instruct_sen[i] for i in motion_idx]
            emotion_instruct_sen = [instruct_sen[i] for i in emotion_idx]
            if len(emotion_instruct_sen):
                emotion_instruct_feat = self.clip_text_encoder.encode_text(emotion_instruct_sen).unsqueeze(1).repeat(1, 77, 1)
            if len(motion_instruct_sen):
                motion_instruct_feat = self.clip_text_encoder.encode(motion_instruct_sen)
            instruct_feat = torch.zeros((batch['audio_flag'].shape[0], 77, self.model_dim), dtype=latent.dtype, device=latent.device)
            if len(motion_instruct_sen):
                instruct_feat = instruct_feat.to(motion_instruct_feat.dtype)
                instruct_feat[motion_idx] = motion_instruct_feat
            if len(emotion_instruct_sen):
                instruct_feat = instruct_feat.to(emotion_instruct_feat.dtype)
                instruct_feat[emotion_idx] = emotion_instruct_feat
  
        
        # get mask
        masks = make_non_pad_mask(batch['latent_length'], xs=batch['latent'][..., 0])
        masks = masks.to(memory_format=torch.contiguous_format)  #  (bs, max_num_frames)
        masks = rearrange(masks, 'b t -> b t 1').contiguous()
        # masks: (bs, T, 1)
        return audio_feat, keylatent, latent, masks, instruct_feat, pose     
