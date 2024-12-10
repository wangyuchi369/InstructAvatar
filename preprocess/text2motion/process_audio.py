import os
import torch
import librosa
import numpy as np
import random
from tqdm import tqdm
import fairseq
import soundfile as sf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=4, help='the number that split the dataset into')
args = parser.parse_args()
wav2vec_path="/xxxx//projects/talkingface_LM/xlsr_53_56k_new.pt"
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vec_path])
model = model[0].cuda()
model.eval()
def process_audio(audio_path, tmpdir, wav2vec_path="/xxxx//projects/talkingface_LM/xlsr_53_56k_new.pt", norm_ref_min=0.3, norm_ref_max=0.4, speed_rate=1.0):
  
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

    wav, sr = sf.read(tmp_audio_path)
    assert sr == 16e3, sr
    # tmp_wav2vec_path = tmp_audio_path.replace(".wav", ".npy")
    person = audio_path.split('/')[-3]
    os.makedirs('/xxxx/TETF/datasets/text2motion/audio_wave2vec_all', exist_ok=True)
    tmp_wav2vec_path = os.path.join('/xxxx/TETF/datasets/text2motion/audio_wave2vec_all', '{}_{}'.format(person, audio_path.split('/')[-1].replace(".wav", ".npy")))
    x = torch.from_numpy(wav).float().cuda()
    with torch.no_grad():
        z = model.feature_extractor(x.unsqueeze(0))
        if isinstance(z, tuple):
            z = z[0]
        z = z.squeeze(0).cpu().numpy()
        np.save(tmp_wav2vec_path, z)
    return tmp_audio_path, tmp_wav2vec_path

input_audio_dir = '/xxxx/TETF/datasets/text2motion/text2express/'
# for person in tqdm(sorted(os.listdir(input_audio_dir))[args.split]):
person = sorted(os.listdir(input_audio_dir))[args.split]
if not person == 'paths.npy':
    
    subdir = os.path.join(input_audio_dir, person, 'audio_16k')
    for audio in tqdm(os.listdir(subdir)):
        audio_path = os.path.join(subdir, audio)
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        empty_audio = np.zeros(int(16000*duration))
        empty_audio_path = audio_path.replace('audio_16k', 'empty_audio')
        os.makedirs(os.path.split(empty_audio_path)[0], exist_ok=True)
        sf.write(empty_audio_path, empty_audio, samplerate=16000)
        if os.path.exists(os.path.join('/xxxx/TETF/datasets/text2motion/audio_wave2vec_all', '{}_{}'.format(person, audio.replace(".wav", ".npy")))):
            continue
        try:
            tmpdir = os.path.join('/xxxx/dataset/tmp', 'tmp')
            tmp_audio_path, tmp_wav2vec_path = process_audio(empty_audio_path, tmpdir)
        except:
            print('error in {}'.format(audio_path))
            continue

    