#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : Tianyu He (xxxx@microsoft.com)
Date               : 2023-05-11 15:07
Last Modified By   : Tianyu He (xxxx@microsoft.com)
Last Modified Date : 2023-05-12 01:59
Description        : 
-------- 
Copyright (c) 2023 Microsoft Corporation.
'''


import os, glob, random
import yaml
import numpy as np
import imghdr
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn


class TalkingCodecLdmkFramesBase(Dataset):
    def __init__(self,
                 data_root,
                 frame_dir,
                 ldmk_dir,
                 img_size=256,
                 load_ldmk=True,
                 load_sameid=False,
                 load_nextframe=False,
                 is_validate_imgs=False,
                 num_people=-1,
                 val_indomain_frames=1000,
                 max_frame_len_each_id=1500,
                 sort_names=True,
                 data_root2=None,
                 data_root3=None):
        """Dataset for HDTF dataset
        Read frames, landmarks from the HDTF dataset.
        Split the dataset into train/val set
            - val: (last id) + (0 - val_indomain_frames frames)
        """
        self.data_root = self._verify_paths(data_root, data_root2, data_root3)

        self.load_ldmk = load_ldmk
        self.load_sameid = load_sameid
        self.load_nextframe = load_nextframe
        self.val_indomain_frames = val_indomain_frames

        # init image preprocessor
        self.preprocessor = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # init global variables
        self.data = list()
        self.total_length = 0
        # read all images and landmarks to validate the data
        self.blacklist = list()
        self._validate_imgs([frame_dir, ldmk_dir], is_validate_imgs)
        # prepare and load the data, prepare for reading in getitem
        self._load(frame_dir, ldmk_dir, num_people, max_frame_len_each_id, sort_names)
        self._prepare()

        assert len(self.data) > 0, f"[TalkingCodecDisentFramesBase][__init__] No data found in {self.data_root}"
        total_minutes = self.total_length / 25. / 60.
        rank_zero_info(f"[TalkingCodecDisentFramesBase][__init__] Loaded {len(self.data)} frames ({total_minutes:.2f} minutes, {total_minutes / 60.:.2f} hours) from {self.data_root}")

    def __len__(self):
        return len(self.data)

    def _verify_paths(self, path1, path2, path3):
        if (path1 is not None) and os.path.isdir(path1):
            return path1
        elif (path2 is not None) and os.path.isdir(path2):
            return path2
        elif (path3 is not None) and os.path.isdir(path3):
            return path3
        else:
            rank_zero_info(f"[TalkingCodecDisentFramesBase] Invalid dirs {path1}, {path2}, {path3}")

    def _prepare(self):
        raise NotImplementedError()

    def _load(self, frame_dir, ldmk_dir, num_people, max_frame_len_each_id, sort_names):
        rank_zero_info(f"[TalkingCodecDisentFramesBase][_load] Loading data from {self.data_root}")
        # random select {num_people} people, -1 means all except the last one
        sub_dirs = sorted(os.listdir(f'{self.data_root}/{frame_dir}'))[:num_people]
        # for each identity
        for sub_dir in tqdm(sub_dirs):
            # random select continuous frames with length {frame_len} from each person
            files = sorted(os.listdir(f'{self.data_root}/{frame_dir}/{sub_dir}'))
            # check if it is a valid directory
            if (len(files) > 0) and os.path.isdir(f'{self.data_root}/{ldmk_dir}/{sub_dir}'):
                if len(files) > max_frame_len_each_id:
                    start = random.randint(0, len(files) - max_frame_len_each_id)
                    files = files[start:start + max_frame_len_each_id]
                # for each data item (each frame)
                for i in range(len(files)):
                    single_data_item = dict()
                    single_data_item['img_path'] = f'{self.data_root}/{frame_dir}/{sub_dir}/{files[i]}'
                    single_data_item['ldmk_path'] = f'{self.data_root}/{ldmk_dir}/{sub_dir}/{files[i]}'
                    # sample img path from the same identity
                    select_pool = list(range(0, i)) + list(range(i+1, len(files)))
                    sameid_i = random.choice(select_pool)
                    single_data_item['img_path_sameid'] = f'{self.data_root}/{frame_dir}/{sub_dir}/{files[sameid_i]}'
                    single_data_item['ldmk_path_sameid'] = f'{self.data_root}/{ldmk_dir}/{sub_dir}/{files[sameid_i]}'
                    # sample img path for next frame
                    if i < len(files) - 1:
                        single_data_item['img_path_nextframe'] = f'{self.data_root}/{frame_dir}/{sub_dir}/{files[i+1]}'
                        single_data_item['ldmk_path_nextframe'] = f'{self.data_root}/{ldmk_dir}/{sub_dir}/{files[i+1]}'
                    else:
                        single_data_item['img_path_nextframe'] = f'{self.data_root}/{frame_dir}/{sub_dir}/{files[i-1]}'
                        single_data_item['ldmk_path_nextframe'] = f'{self.data_root}/{ldmk_dir}/{sub_dir}/{files[i-1]}'

                    # validate the data item
                    valid = True
                    valid = valid and (single_data_item['img_path'] not in self.blacklist)
                    if self.load_ldmk:
                        valid = valid and (single_data_item['ldmk_path'] not in self.blacklist)
                    if self.load_sameid:
                        valid = valid and (single_data_item['img_path_sameid'] not in self.blacklist)
                        if self.load_ldmk:
                            valid = valid and (single_data_item['ldmk_path_sameid'] not in self.blacklist)
                    if self.load_nextframe:
                        valid = valid and (single_data_item['img_path_nextframe'] not in self.blacklist)
                        if self.load_ldmk:
                            valid = valid and (single_data_item['ldmk_path_nextframe'] not in self.blacklist)

                    if valid:
                        # add to data list
                        self.total_length += 1
                        self.data.append(single_data_item)
        if sort_names:
            self.data = sorted(self.data, key=lambda x: x['img_path'])
        rank_zero_info(f"[TalkingCodecDisentFramesBase][_load] Loaded data with {len(self.data)} frames with keys: image_sameid ({self.load_sameid}), image_nextframe ({self.load_nextframe}), ldmk ({self.load_ldmk})")

    def _validate_imgs(self, dirs, is_validate_imgs):
        if len(self.blacklist) > 0:
            for blacklist in self.blacklist:
                self.blacklist.remove(blacklist)
                self.blacklist.append(f'{self.data_root}/{blacklist}')
                rank_zero_info(f"[TalkingCodecDisentFramesBase][_validate_imgs] Invalid image {self.data_root}/{blacklist}")
        if is_validate_imgs:
            for dir in dirs:
                rank_zero_info(f"[TalkingCodecDisentFramesBase][_validate_imgs] Validating images in {self.data_root}/{dir}")
                # e.g., frames, ldmks
                for sub_dir in tqdm(os.listdir(f'{self.data_root}/{dir}')):
                    # e.g., 0001, 0002
                    for filename in os.listdir(f'{self.data_root}/{dir}/{sub_dir}'):
                        # e.g., 0001_0001.jpg
                        if not self._validate_img(f'{self.data_root}/{dir}/{sub_dir}/{filename}'):
                            rank_zero_info(f"[TalkingCodecDisentFramesBase][_validate_imgs] Invalid image {self.data_root}/{dir}/{sub_dir}/{filename}")
                            self.blacklist.append(f'{self.data_root}/{dir}/{sub_dir}/{filename}')

    def _validate_img(self, img_path):
        try:
            # img = Image.open(img_path)
            # img.verify()
            img_format = imghdr.what(img_path)
            if (img_format is None) or (img_format not in ['jpeg', 'png']):
                return False
            return True
        except:
            return False

    def _preprocess_img(self, img_path, target_format="RGB"):
        img = Image.open(img_path)
        if not img.mode == target_format:
            img = img.convert(target_format)
        img = self.preprocessor(img)
        return img

    def __getitem__(self, i):
        # init data item
        data_item = {'index': i}
        data_item['filename'] = os.path.basename(self.data[i]['img_path']).split('.')[0]
        try:
            # load image
            data_item['image'] = self._preprocess_img(self.data[i]['img_path'])
            if self.load_ldmk:
                data_item['ldmk'] = self._preprocess_img(self.data[i]['ldmk_path'], target_format="RGB")
            # load sameid image
            if self.load_sameid:
                data_item['image_sameid'] = self._preprocess_img(self.data[i]['img_path_sameid'])
                if self.load_ldmk:
                    data_item['ldmk_sameid'] = self._preprocess_img(self.data[i]['ldmk_path_sameid'], target_format="RGB")
            # load next frame image
            if self.load_nextframe:
                data_item['image_nextframe'] = self._preprocess_img(self.data[i]['img_path_nextframe'])
                if self.load_ldmk:
                    data_item['ldmk_nextframe'] = self._preprocess_img(self.data[i]['ldmk_path_nextframe'], target_format="RGB")
        except:
            rank_zero_info(f"[TalkingCodecDisentFramesBase][__getitem__] Failed to load image {self.data[i]['img_path']} or landmark {self.data[i]['ldmk_path']}, poping it from the list")
            self.data.pop(i)
            if i >= len(self.data):
                i = len(self.data) - 1
            return self.__getitem__(i)
        return data_item


class TalkingCodecLdmkFramesTrain(TalkingCodecLdmkFramesBase):
    def _prepare(self):
        rank_zero_info(f"[TalkingCodecLdmkFramesTrain] Spliting in-domain training data from {self.val_indomain_frames} to end")
        self.data = self.data[self.val_indomain_frames:]


class TalkingCodecLdmkFramesValidation(TalkingCodecLdmkFramesBase):
    def _prepare(self):
        rank_zero_info(f"[TalkingCodecLdmkFramesValidation] Spliting in-domain validation data from 0 to {self.val_indomain_frames}")
        self.data = self.data[:self.val_indomain_frames]