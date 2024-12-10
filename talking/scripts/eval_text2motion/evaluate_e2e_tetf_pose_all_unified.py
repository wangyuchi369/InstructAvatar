#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : Tianyu He (xxxx@microsoft.com)
Date               : 2023-06-17 14:38
Last Modified By   : Tianyu He (xxxx@microsoft.com)
Last Modified Date : 2023-08-07 09:54
Description        : 
-------- 
Copyright (c) 2023 Microsoft Corporation.
'''

"""evaluete end-to-end with both codec and a2e model"""


import argparse, os, sys, shutil
import random
sys.path.append('~/TETF')
from glob import glob
import PIL
import torch
import pickle
import json
import numpy as np
import cv2
from omegaconf import OmegaConf
from moviepy.editor import ImageSequenceClip, clips_array, AudioFileClip
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from scipy.signal import convolve
import librosa
from talking.modules.a2vqvae.clip import FrozenCLIPEmbedder
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from torch import autocast
import soundfile as sf
import importlib
from contextlib import nullcontext
import time
from lightning.pytorch import seed_everything

from ldm.util import instantiate_from_config
from talking.scripts.get_landmark import regress_landmarks
import warnings
warnings.filterwarnings("ignore")

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    print(f"Loaded model from {ckpt}")
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model


def load_img(img_path):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(256),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(img_path).convert('RGB')
    # ldmk_img = Image.open(img_path.replace("frames", "ldmks")).convert('RGB')
    # ldmk_img = transform(ldmk_img)
    img = transform(img)
    return img


def load_ldmk(ldmk_img):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(256),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ldmk_img = ldmk_img.convert('RGB')
    ldmk_img = transform(ldmk_img)
    return ldmk_img


def draw_kp(kp, size=(256,256), is_connect=False, color=(255,255,255)):
    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for i in range(kp.shape[0]):
        x = int((kp[i][0]))
        y = int((kp[i][1]))
        thinkness = 1 if is_connect else 1
        frame = cv2.circle(frame, (x, y), thinkness, color, -1)
    return frame


def process_audio(audio_path, tmpdir, wav2vec_path="xlsr_53_56k_new.pt", norm_ref_min=0.3, norm_ref_max=0.4, speed_rate=1.0):
    import fairseq
    import soundfile as sf
    # convert to 16kHz, single channel
    os.makedirs(tmpdir, exist_ok=True)
    cmd_to16kHz = f"sox {audio_path} -r 16000 -c 1 {os.path.join(tmpdir, os.path.split(audio_path)[1])}"
    os.system(cmd_to16kHz)
    tmp_audio_path = os.path.join(tmpdir, os.path.split(audio_path)[1])
    # normalize
    wav, _ = librosa.load(tmp_audio_path, sr=16000)
    # adjust the speeed rate
    if speed_rate != 1.0:
        wav = librosa.effects.time_stretch(wav, rate=speed_rate)  # time stretch, rate < 1.0 for slow
    norm_wav = np.linalg.norm(wav, ord=np.inf)
    normed_wav = wav * random.uniform(norm_ref_min, norm_ref_max) / norm_wav
    tmp_audio_path = tmp_audio_path.replace(".wav", "_normed.wav")
    sf.write(tmp_audio_path, normed_wav, samplerate=16000)
    # extract wav2vec
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vec_path])
    model = model[0].cuda()
    model.eval()
    wav, sr = sf.read(tmp_audio_path)
    assert sr == 16e3, sr
    tmp_wav2vec_path = tmp_audio_path.replace(".wav", ".npy")
    x = torch.from_numpy(wav).float().cuda()
    with torch.no_grad():
        z = model.feature_extractor(x.unsqueeze(0))
        if isinstance(z, tuple):
            z = z[0]
        z = z.squeeze(0).cpu().numpy()
        np.save(tmp_wav2vec_path, z)
    return tmp_audio_path, tmp_wav2vec_path


def moving_average(x, w=5):
    assert w % 2 == 1, "w should be single!"
    kernel = np.ones(tuple([w] + [1] * (x.ndim - 1)))
    a = convolve(x, kernel, 'valid') / w
    return np.vstack((a[0:1, :].repeat(w//2, axis=0), a, a[-1:, :].repeat(w//2, axis=0)))


def get_mouth_unrelated_index():
    # remove the index of mouth area
    mouth_index = list(set([1, 122, 657, 202, 204, 43, 209, 218, 60, 61, 219, 220, 63, 221, 64, 
                            65, 636, 69, 637, 222, 66, 67, 223, 68, 224, 225, 70, 226, 227, 74, 
                            71, 75, 228, 72, 73, 660, 88, 316, 436, 516, 518, 358, 523, 532, 375, 
                            376, 533, 534, 378, 535, 379, 380, 384, 536, 381, 382, 537, 383, 538, 
                            539, 385, 540, 541, 386, 389, 542, 387, 388, 640]))
    index = list(range(669))
    for i in mouth_index:
        index.remove(i)
    return index


def main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--codec_config",
        type=str,
        # default="configs/vqvae_2enc-hdtf/test_autoencoder_seperate_kl_64x64x3.yaml",
        default="configs/vqvae_2enc-hdtf/tetf_autoencoder_scale_avena_extend_dense_64x64x3.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--codec_ckpt",
        type=str,
        # default="/data/datasets/HDTF/clean_videos_split10s_ptcode/latents-seperate-vae-split10s-ptcode-set2-515/seperate_ldmk_hdtf_set2_last_515.ckpt",
        #default="/xxxx//diffusion/stable-diffusion/latents/seperate_ldmk_ccd_set2_ep234.ckpt",
        # default="/data/datasets/AVENA_CCD_HDTF/latents-700m-avena-ep23/seperate_ldmk_scale_avena_dense_vae_set2_ep23.ckpt",
        default="/xxxx//diffusion/stable-diffusion/latents/seperate_ldmk_scale_avena_dense_vae_set2_randft_ep12.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--a2e_config",
        type=str,
        default="configs/a2vqvae_conformer-htdf/var_tetf_ep234_ada_conformer_12l_keymo_invslr.yaml",
        # default="configs/a2vqvae-hdtf/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--a2e_ckpt",
        type=str,
        default="amlt/hdtf_diff_condconfo12l_keymo_ep23_invslr-lr1-4/hdtf_diff_condconfo12l_keymo_ep23_invslr-lr1-4/2023-07-15T12-28-03_hdtf_diff_condconfo12l_keymo_ep23_invslr/checkpoints/epoch=004999.ckpt",
        # default="amlt/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc_default250len-lr-7/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc_default250len-lr-7/2023-06-12T14-48-31_hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc/checkpoints/epoch=005499.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--a2e_predict_ldmk",
        type=str2bool,
        const=True,
        default=False,
        # default=True,
        nargs="?",
        help="a landmark a2e model",
    )
    parser.add_argument(
        "--a2e_smooth_ldmk",
        type=str2bool,
        const=True,
        # default=False,
        default=True,
        nargs="?",
        help="a landmark a2e model with smoothed landmark",
    )
    parser.add_argument(
        "--a2e_only_recons_video",
        type=str2bool,
        const=True,
        default=False,
        # default=True,
        nargs="?",
        help="only save reconstructed video",
    )
    parser.add_argument(
        "--a2e_use_both_keymo_keyappear",
        type=str2bool,
        const=True,
        default=False,
        # default=True,
        nargs="?",
        help="a2e model is conditioned on both keymo and keyappear",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=250,
        help="max_inference_length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--audio_feat",
        type=str,
        # default="/data/datasets/HDTF/clean_videos_split10s_ptcode/audios_16k_wav2vec/WRA_TimScott_000-00000.npy",
        # default="../datasets/HDTF/clean_videos_split10s_ptcode/audios_16k_wav2vec/WRA_LamarAlexander0_000-00000.npy",
        # default="demo_20230728/audios_3rd_16k_single_normed_wav2vec/00022.npy",
        default="",
        help="path to the test data, must contains `filenames.pickle`",
    )
    parser.add_argument(
        "--appear_dir",
        type=str,
        default="demo_20230728/appear/vicky",
        help="path to the the appearance image",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        # default="/data/datasets/HDTF/clean_videos_split10s_ptcode/audios_16k/WRA_TimScott_000-00000.wav",
        # default="/data/datasets/HDTF/clean_videos_split10s_ptcode/audios_16k/WRA_LamarAlexander0_000-00000.wav",
        default="/xxxx/dataset/text2motion/MEAD_clean/M003/audio_16k/happy_011_0.wav",
        help="audio path",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="amlt/hdtf_diff_condconfo12l_keymo_ep23_invslr-lr1-4/hdtf_diff_condconfo12l_keymo_ep23_invslr-lr1-4/2023-07-15T12-28-03_hdtf_diff_condconfo12l_keymo_ep23_invslr/videos_benchmark/appear-vicky_audio-00022_epoch-004999.mp4",
        # default="amlt/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc_default250len-lr-7/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc_default250len-lr-7/2023-06-12T14-48-31_hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc/videos_benchmark/appear-20160226VIDEIIu7qJbgwCqew7w6XCJA160226_audio-WRA_TimScott_000-00000_epoch-005499.mp4",
        help="name of the saved video",
    )
    parser.add_argument(
        "--audio_save_dir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="tmp/audio_save_dir",
    )
    parser.add_argument(
        "--ldmks_save_dir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="tmp/ldmks_save_dir",
    )
    parser.add_argument(
        "--imgs_save_dir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="tmp/imgs_save_dir",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="smile",
        nargs="+",
        help="the instruction to condition on",
    )
    parser.add_argument(
        "--audio_flag",
        type=str2bool,
        default=True,
        help="whether to provide audio"
    )
    parser.add_argument(
        "--inst_flag",
        type=str2bool,
        default=True,
        help="whether to provide instruction"
    )
    parser.add_argument(
        "--t2m_length",
        type=int,
        default=225,
        help="the length of the text2motion case"
    )
    parser.add_argument(
        "--use_eos",
        type=str2bool,
        default=False,
        help="whether to use eos"
    )
    
    t2m_data = json.load(open('t2m_test_data.json'))
    # t2m_test_data = random.sample(t2m_data.keys(), 100)
    
    
    opt = parser.parse_args()
    seed_everything(opt.seed)
    audio_sample_rate = 16000
    
    # load models
    codec_config = OmegaConf.load(f"{opt.codec_config}")
    a2e_config = OmegaConf.load(f"{opt.a2e_config}")
    print(f"Evaluating model {opt.codec_ckpt} & {opt.a2e_ckpt}")
    codec_model = load_model_from_config(codec_config, f"{opt.codec_ckpt}")
    a2e_model = load_model_from_config(a2e_config, f"{opt.a2e_ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    codec_model = codec_model.to(device)
    a2e_model = a2e_model.to(device)
    
    
    for each_t2m in t2m_data.keys():
    # text2motion case
        if not opt.audio_flag:
            print(int(audio_sample_rate * opt.t2m_length / 25))
            samples = np.zeros(int(audio_sample_rate * opt.t2m_length / 25))
            # 写入wav文件
            opt.audio_path = os.path.join(opt.audio_save_dir, 't2m', 'tmp_for_t2m.wav')
            os.makedirs(os.path.dirname(opt.audio_path), exist_ok=True)
            print(samples)
            print(opt.audio_path)
            print(samples.shape)
            sf.write(opt.audio_path, samples, audio_sample_rate)
        if opt.audio_feat == '':
            print(f"Extracting audio features from {opt.audio_path}")
            tmp_audio_path, tmp_wav2vec_path = process_audio(opt.audio_path, opt.audio_save_dir)
            opt.audio_path = tmp_audio_path
            opt.audio_feat = tmp_wav2vec_path
            torch.cuda.empty_cache()



        # read audio feat
        audio_feat = np.load(opt.audio_feat).T[:2*(opt.max_len-25)]  # (frames, 512), frames may exceed max_len during training
        audio_feat_len = audio_feat.shape[0]
        
        if not opt.inst_flag:
            opt.instrictoon = random.choice(['talk with free expression.', 'talk with neutral expression', 'talk naturally'] )
        
        # opt.instruction = t2m_data[each_t2m]
        opt.instruction = random.choice(list(t2m_data.values()))
        
        opt.video_name = f'/mnt/blob/xxxx/TETF/text2motion/unified_rand/{each_t2m}.mp4'
        instruction = opt.instruction
        clip_text_embedder = FrozenCLIPEmbedder().to(device)
        if opt.use_eos and opt.audio_flag:
            instruct_feat = clip_text_embedder.encode_text(instruction).to(device).unsqueeze(1).repeat(1, 77, 1)
        else:
            instruct_feat = clip_text_embedder.encode(instruction).to(device)
        
        person, video_t2m_name = "_".join(each_t2m.split('_')[:4]), '_'.join(each_t2m.split('_')[4:])
        tmp_video = cv2.VideoCapture(f'/mnt/blob/xxxx/TETF/datasets/text2motion/text2express/{person}/video_crop_resize_25fps_16k/{video_t2m_name}.MP4')
        ret, frame = tmp_video.read()
        if ret:
            os.makedirs(f'/mnt/blob/xxxx/TETF/datasets/text2motion/first_frame', exist_ok=True)
            cv2.imwrite(f'/mnt/blob/xxxx/TETF/datasets/text2motion/first_frame/{each_t2m}.jpg', frame)
        else:
            print(f'Error: {each_t2m} cannot read the first frame')
            continue
        tmp_video.release()
        opt.appear_dir = f'/mnt/blob/xxxx/TETF/datasets/text2motion/first_frame/{each_t2m}.jpg'
        # read appear img
        appear_img_path = opt.appear_dir
        # appear_img_path = appear_img_path[random.randrange(0, len(appear_img_path))]
        appear_rgb = load_img(appear_img_path)
        if len(appear_rgb.shape) == 3:
            appear_rgb = appear_rgb[None, ...]
            appear_rgb = appear_rgb.to(memory_format=torch.contiguous_format).float()
        # extract ldmk from the appear img
        img_256 = Image.open(appear_img_path).convert('RGB').resize((256,256))
        ldmk_256, ldmk_sigma = regress_landmarks.regress_landmarks_demo(np.asarray(img_256)/255.)  # get all the coordinates
        ldmk_img = Image.fromarray(draw_kp(ldmk_256, size=(256,256), is_connect=False, color=(255,255,255)))
        ldmk_img = load_ldmk(ldmk_img)

        # put all input to device
        appear_rgb = appear_rgb.to(device)
        audio_feat_paded = torch.zeros((opt.max_len * 2, 512), dtype=torch.float32)
        audio_feat_paded[:audio_feat.shape[0]] = torch.from_numpy(audio_feat)
        audio_feat = audio_feat_paded[None, ...].to(device)
        ldmk_256 = torch.from_numpy(ldmk_256)[None, ...].to(device)
        ldmk_img = ldmk_img[None, ...].to(device)

        if not os.path.exists(opt.imgs_save_dir):
            os.makedirs(opt.imgs_save_dir, exist_ok=True)
        else:
            shutil.rmtree(opt.imgs_save_dir)
            os.makedirs(opt.imgs_save_dir, exist_ok=True)
        
        if opt.a2e_predict_ldmk:
            if not os.path.exists(opt.ldmks_save_dir):
                os.makedirs(opt.ldmks_save_dir, exist_ok=True)
            else:
                shutil.rmtree(opt.ldmks_save_dir)
                os.makedirs(opt.ldmks_save_dir, exist_ok=True)

        if not os.path.exists(os.path.dirname(opt.video_name)):
            os.makedirs(os.path.dirname(opt.video_name), exist_ok=True)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with precision_scope("cuda"):
            with torch.no_grad():
                tic = time.time()

                z_ldmk, ldmk_middle, ldmk_posterior = codec_model.encode_ldmk(ldmk_img, sample_posterior=False)
                if opt.a2e_predict_ldmk:
                    if opt.a2e_use_both_keymo_keyappear:
                        raise NotImplementedError
                    else:
                        if a2e_config.data.params.train.params.keyla_key == 'appear_latent':
                            appear_posterior, appear_h = codec_model.encode(appear_rgb.repeat(1, 1, 1, 1))
                            z_appear = appear_posterior.mode()
                            ldmks = a2e_model.evaluate(audio_feat=audio_feat, keylatent=z_appear.reshape((1, 1, 12288)), inst_feat=instruct_feat,diff_init_shape=1338)
                        elif a2e_config.data.params.train.params.keyla_key == 'motion_latent' or a2e_config.data.params.train.params.keyla_key == 'ldmk':
                            ldmk_256 = ldmk_256.to(dtype=audio_feat.dtype)
                            ldmks = a2e_model.evaluate(audio_feat=audio_feat, keylatent=ldmk_256.reshape((1, 1, 1338)), inst_feat=instruct_feat,diff_init_shape=1338)  # (1, T, 1338)
                        else:
                            raise Exception(f'keyla_key {a2e_config.data.params.train.params.keyla_key} not supported')
                        if opt.a2e_smooth_ldmk:
                            # ldmk: (1, T, 1338)
                            print(f'Generating with smoothed ldmk')
                            assert ldmks.shape[0] == 1, f'{ldmks.shape[0]} != 1'
                            ldmks_np = ldmks.cpu().numpy().copy().reshape((ldmks.shape[1], 669, 2))
                            index = get_mouth_unrelated_index()
                            ldmks_np[:, index, :] = moving_average(ldmks_np[:, index, :], w=5)
                            ldmks = torch.from_numpy(ldmks_np).to(device).reshape((1, ldmks.shape[1], 1338))
                    seq_len = ldmks.shape[1]
                else:
                    if opt.a2e_use_both_keymo_keyappear:
                        appear_posterior, appear_h = codec_model.encode(appear_rgb.repeat(1, 1, 1, 1))
                        z_appear = appear_posterior.mode()
                        motion_latent = a2e_model.evaluate(audio_feat=audio_feat, keymo=ldmk_middle.reshape((1, 1, 768)), keyappear=z_appear.reshape((1, 1, 12288)),inst_feat=instruct_feat, diff_init_shape=768)
                    else:
                        motion_latent = a2e_model.evaluate(audio_feat=audio_feat, keylatent=ldmk_middle.reshape((1, 1, 768)), inst_feat=instruct_feat, diff_init_shape=768, audio_flag=opt.audio_flag, inst_flag=opt.inst_flag)  # (1, T, 768)
                        # print('here!')
                        # motion_latent = torch.from_numpy(np.load('/xxxx/TETF/datasets/text2motion/text2motion_latents_700M_ep12/motion_CC_part_10_1_1142_01_8_0.npy')).reshape(1,-1,768).to(device)
                        # # motion_latent = torch.from_numpy(np.load('/xxxx/TETF/datasets/MEAD/MEAD_latents_700M_ep12/motion_M003_neutral_006_0.npy')).reshape(1,-1,768).to(device)
                    seq_len = motion_latent.shape[1]

                seq_len = min(seq_len, audio_feat_len // 2)
                for idx in tqdm(range(seq_len)):
                    if opt.a2e_predict_ldmk:
                        pred_ldmk = ldmks[:, idx, :].reshape((669, 2))
                        pred_ldmk_img = Image.fromarray(draw_kp(pred_ldmk, size=(256,256), is_connect=False, color=(255,255,255)))
                        pred_ldmk_img.save(os.path.join(opt.ldmks_save_dir, "ldmkspred_%08d.png" % idx))
                        pred_ldmk_img = load_ldmk(pred_ldmk_img)[None, ...].to(device)
                        reconstructions, _, _, _, _ = codec_model.forward(appear_rgb.repeat(1, 1, 1, 1), 
                                                                        ldmks=pred_ldmk_img, 
                                                                        concat_input=False,
                                                                        sample_posterior=False)
                    else:
                        cond_motion = motion_latent[:, idx, :].reshape((-1,3,16,16))
                        reconstructions = codec_model.forward_fromldmklatent(appear_img=appear_rgb.repeat(1, 1, 1, 1), 
                                                                            ldmk_middle=cond_motion, 
                                                                            concat_input=False,
                                                                            sample_posterior=False)
                    reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0)
                    recons = reconstructions.detach().cpu()
                    grid = make_grid(recons, nrow=4)
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid * 255).astype(np.uint8)
                    img = Image.fromarray(grid)
                    img.save(os.path.join(opt.imgs_save_dir, "recons_%08d.png" % idx))
                toc = time.time()

        tmp_audio_duration = min(8, librosa.get_duration(path=opt.audio_path))
        # appear_video = ImageSequenceClip(opt.appear_dir, fps=25).set_duration(tmp_audio_duration)
        recon_video = ImageSequenceClip(opt.imgs_save_dir, fps=25).set_duration(tmp_audio_duration)

        # appear_video = appear_video.resize(recon_video.size)
        final_video = clips_array([[recon_video]])
        if opt.a2e_predict_ldmk:
            ldmk_video = ImageSequenceClip(opt.ldmks_save_dir, fps=25).set_duration(tmp_audio_duration)
            ldmk_video = ldmk_video.resize(recon_video.size)
            final_video = clips_array([ ldmk_video, recon_video])

        if opt.a2e_only_recons_video:
            final_video = recon_video
        if opt.audio_flag:
            final_video.audio = AudioFileClip(opt.audio_path, fps=16000).set_duration(tmp_audio_duration)
        final_video.write_videofile(opt.video_name)
        print(f"Output to: {opt.video_name}. Time taken: {toc - tic:.2f} seconds.")


if __name__ == "__main__":
    main()
