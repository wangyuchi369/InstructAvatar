import sys
import os
import torch
from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
import numpy as np
import argparse
import cv2

from torch import autocast
from ldm.util import instantiate_from_config
from talking.scripts.get_landmark import regress_landmarks, draw_kp

class ValidVideoDataset(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        video = cv2.VideoCapture(img_path)
        self.frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            self.frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(256),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.transform(self.frames[idx])

        img_256 = self.frames[idx].resize((256,256))
        dense_img_256, sigma_img = regress_landmarks.regress_landmarks_demo(np.asarray(img_256)/255.)
        gray_img = Image.fromarray(draw_kp(dense_img_256, size=(256,256), is_connect=False, color=(255,255,255)))
        gray_img = self.transform(gray_img)
        return {'img': img, 'gray_img': gray_img, 'index': idx, 'ldmk': dense_img_256}

class ValidDataset(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(256),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.img_path_list = sorted(glob(f'{self.img_path}/*'))
        self.suffix = [image.split("/")[-1] for image in self.img_path_list]

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_path_list[idx])
        # if the image is 'rgba'!
        img = img.convert('RGB')
        img = self.transform(img)

        img_256 = Image.open(self.img_path_list[idx]).convert('RGB').resize((256,256))
        dense_img_256, sigma_img = regress_landmarks.regress_landmarks_demo(np.asarray(img_256)/255.)
        gray_img = Image.fromarray(draw_kp(dense_img_256, size=(256,256), is_connect=False, color=(255,255,255)))
        gray_img = self.transform(gray_img)
        return {'img': img, 'gray_img': gray_img, 'suffix': self.suffix[idx], 'index': idx, 'ldmk': dense_img_256}

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, help='ckpt path')
parser.add_argument('--config', type=str, help='config of the model')
parser.add_argument('--folder_path', type=str, help='the folder of hdtf')
parser.add_argument('--save_path', type=str, help='path to save latent codes')
parser.add_argument('--bsz', type=int, default=12, help='batch size')
parser.add_argument('--split', type=int, default=4, help='the number that split the dataset into')
parser.add_argument('--num', type=int, default=0, help='the start number of the split')
parser.add_argument('--ccd', action='store_true', help='process ccd or not. default is hdtf')
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
device = 'cuda:0'
config = OmegaConf.load(f"{args.config}")

print(f"Evaluating model {args.ckpt}")
model = load_model_from_config(config, f"{args.ckpt}")
model = model.to(device)

if args.ccd:
    orig_paths = list(np.load('/mnt/blob/v-leyili/dataset/CCD/CCDv1_processed/paths_t2e.npy', allow_pickle=True))
    paths = orig_paths[args.num::args.split]
else:
    paths = sorted(os.listdir(args.folder_path))[args.num::args.split]

for sub_dir in tqdm(paths):
    person = sub_dir.split('/')[-3]
    cc_sub_dir = '{}_{}'.format(person, sub_dir.split('/')[-1].split('.')[0])
    if os.path.exists(f'{args.save_path}/ldmk_{cc_sub_dir}.npy') and \
        os.path.exists(f'{args.save_path}/appear_{cc_sub_dir}.npy') and \
        os.path.exists(f'{args.save_path}/motion_{cc_sub_dir}.npy'):
        continue
    

    if args.ccd:
        val_dataloader = DataLoader(ValidVideoDataset(sub_dir), 
                                batch_size=args.bsz, shuffle=False, num_workers=5)
    else:
        val_dataloader = DataLoader(ValidDataset(os.path.join(args.folder_path, sub_dir)), 
                                batch_size=args.bsz, shuffle=False, num_workers=5)

    all_appear_cond = []
    all_motion_cond = []
    all_ldmks = []
    precision_scope = autocast
    with precision_scope("cuda"):
        for samples in tqdm(val_dataloader, desc="val data"):
            input = model._get_input(samples, 'img').to(device)
            input_ldmk = model._get_input(samples, 'gray_img').to(device)
            ldmk = samples['ldmk']
            with torch.no_grad():
                appear_posterior, appear_middle = model.encode(input)
                z_appear = appear_posterior.mode()
                _, cond_motion, _ = model.encode_ldmk(input_ldmk, sample_posterior=False)
            all_ldmks.append(ldmk.cpu())
            all_appear_cond.append(z_appear.cpu())
            all_motion_cond.append(cond_motion.cpu())

    if all_ldmks == [] or all_appear_cond == [] or all_motion_cond == []:
        continue
    all_appear_cond = torch.cat(all_appear_cond, dim=0).contiguous()
    all_motion_cond = torch.cat(all_motion_cond, dim=0).contiguous()
    all_ldmks = torch.cat(all_ldmks, dim=0).contiguous()
    if args.ccd:
        person = sub_dir.split('/')[-3]
        sub_dir = '{}_{}'.format(person, sub_dir.split('/')[-1].split('.')[0])
    np.save(f'{args.save_path}/ldmk_{sub_dir}.npy', all_ldmks.detach().cpu().numpy())
    np.save(f'{args.save_path}/appear_{sub_dir}.npy', all_appear_cond.detach().cpu().numpy())
    np.save(f'{args.save_path}/motion_{sub_dir}.npy', all_motion_cond.detach().cpu().numpy())