
import argparse
import glob
import os
from shutil import copy

import numpy as np
import soundfile as sf
import torch
import tqdm
import fairseq
from torch import nn

# PYTHONPATH=/data/talking-diffae-tianyu/fairseq
# hydra-core==1.2.0

# https://github.com/facebookresearch/fairseq/issues/4585
fname = "xlsr_53_56k_new.pt"
# dir_wav = "/data/datasets/HDTF/clean_videos_split10s_ptcode/audios"
dir_wav_16k = "/data/datasets/CCD/CCDv1_processed/clean_merge/audio_16k_single"
dir_wav2vec = "/data/datasets/CCD/CCDv1_processed/clean_merge/audio_16k_single_wav2vec"


# os.makedirs(dir_wav_16k, exist_ok=True)
# for wav_file in os.listdir(dir_wav):
#     path_a = os.path.join(dir_wav, wav_file)
#     cmd_to16kHz = f"sox {path_a} -r 16000 -c 1 {os.path.join(dir_wav_16k, os.path.split(path_a)[1])}"
#     os.system(cmd_to16kHz)


def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """
    wav, sr = sf.read(fname)
    assert sr == 16e3, sr
    return wav, 16e3


model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([fname])
model = model[0].cuda()
model.eval()


os.makedirs(dir_wav2vec, exist_ok=True)
for name in tqdm.tqdm(glob.glob(os.path.join(dir_wav_16k, "*.wav"))):
    wav, sr = read_audio(name)
    x = torch.from_numpy(wav).float().cuda()
    with torch.no_grad():
        try:
            z = model.feature_extractor(x.unsqueeze(0))
            if isinstance(z, tuple):
                z = z[0]
            z = z.squeeze(0).cpu().numpy()
            np.save(os.path.join(dir_wav2vec, os.path.split(name)[1].replace(".wav", ".npy")), z)
        except:
            print('dropping:', name)
            continue