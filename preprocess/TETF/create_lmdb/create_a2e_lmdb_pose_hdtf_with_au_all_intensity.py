#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, glob
import lmdb
import pickle
import pyarrow
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
from transformers import CLIPProcessor, CLIPModel
from talking.modules.a2vqvae.clip import FrozenCLIPEmbedder
import json
import re
import sys

sys.path.append(os.getcwd())



def check_dir(*path):
    """Check dir(s) exist or not, if not make one(them).
    Args:
        path: full path(s) to check.
    """
    for p in path:
        os.makedirs(p, exist_ok=True)


def get_key(index):
    return u'{}'.format(index).encode('ascii')


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    return pyarrow.serialize(obj).to_buffer()


class TalkingA2VqvaeSeqKeylaBase(Dataset):
    def __init__(self,
                 data_root,
                 mead_latent_dir_name,
                 mead_audio_feat_dir_name,
                 mead_pose_dir_name,
                 mead_action_units_dir_name,
                #  ccdv2_audio_feat_dir_name,
                 hdtf_latent_dir_name,
                 hdtf_audio_feat_dir_name,
                 hdtf_pose_dir_name,
                 t2m_latent_dir_name,
                 min_num_frames=25,
                 target_motion_dim=768,
                 target_appear_dim=12288,
                 target_ldmk_dim=1338,
                 target_pose_dim=3,
                 audio_feat_dim=512):
        self.data_root = data_root
        self.min_num_frames = min_num_frames
        self.target_motion_dim = target_motion_dim
        self.target_appear_dim = target_appear_dim
        self.target_ldmk_dim = target_ldmk_dim
        self.audio_feat_dim = audio_feat_dim
        self.target_pose_dim = target_pose_dim
        # self.clip_text_embedder = FrozenCLIPEmbedder()
        self.data = list()
        self._load_hdtf(hdtf_latent_dir_name,
                        hdtf_audio_feat_dir_name,
                        hdtf_pose_dir_name)
        # self._load_mead(mead_latent_dir_name,
        #                mead_audio_feat_dir_name, mead_pose_dir_name, mead_action_units_dir_name)
        # self._load_t2m(t2m_latent_dir_name)
        self.last_aligned_latent_len, self.last_motion_latent, self.last_ldmk, self.last_appear_latent, self.last_audio_feat, self.last_filename, self.instruction, self.last_pose, self.inst_flag, self.audio_flag, self.action_unit = None, None, None, None, None, None, None, None, None, None, None

        assert len(self.data) > 0, f"[TalkingA2VqvaeSeqKeylaBase] No data found in {self.data_root}"

    def __len__(self):
        return len(self.data)

    def _check_instruction(self, instruction):
        output_string = re.sub(r'[^\w,!]+', '', instruction)
        if not output_string:
            return False
        expected_list = ['smile', 'angry', 'sad', 'surprise', 'disgust', 'fear', 'neutral', "natural", "happy", "calm", "angry", "sad", "surprised", "disgusted", "fearful",
                         "disappointed", "bored", "excited", "frustrated", "proud", "ashamed", "amused", "excited", "tired", "sleepy", "neutral", "natural", "happy", "calm",
                         "open", "close", "turn", "move", "shake", "nod", "tilt", "rotate", "lift", "lower", "push", "pull", "wave", "point", "grasp", "release", "squeeze","random",
                         "suprised", "free", "annoyed", "eyes", "mouth", "laugh", "smile", "cry", "scream", "yell", "talk", "whisper", "sing", "shout", "speak", "squeak", "squeal",
                         "scared","confused","smile", "frown", "laugh", "wink", "surprise", "anger", "sadness", 
    "disgust", "fear", "confusion", "excitement", "happiness", "contentment", 
    "disappointment", "embarrassment","nod", "shake head", "tilt head", "turn head", "bob head", 
    "raise eyebrows", "scrunch eyebrows", "roll eyes", "head bang", 
    "head tilt", "head shake", "head nod", "head turn", "head tilt","head","eyebrows","terrified","sleepy",
    "silly","grumpy","mean","curious","frustrated","proud","freestyle","smirk","chin","relax","interested"]
        for word in expected_list:
            if word in output_string:
                return True
        return False
  
    def _check_t2m_latent(self,
                      motion_latent,
                      appear_latent,
                      ldmk,
                      file_name,
                      threshold=2):
        if (motion_latent is None) or (appear_latent is None) or (ldmk is None):
            return False
        aligned_latent_len = min( motion_latent.shape[0], appear_latent.shape[0], ldmk.shape[0])
        if aligned_latent_len < self.min_num_frames:
            return False
        if abs(motion_latent.shape[0] - aligned_latent_len) > threshold:
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: motion_latent_len: {motion_latent.shape[0]}, , appear_latent_len: {appear_latent.shape[0]}")
            return False
        if abs(ldmk.shape[0] - aligned_latent_len) > threshold:
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: motion_latent_len: {motion_latent.shape[0]}, ldmk_len: {ldmk.shape[0]}, appear_latent_len: {appear_latent.shape[0]}, ")
        #     return False
        if abs(appear_latent.shape[0] - aligned_latent_len) > threshold:
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: motion_latent_len: {motion_latent.shape[0]}, appear_latent_len: {appear_latent.shape[0]}, ")
            return False
        # if abs(audio_feat.shape[0] - aligned_latent_len * 2) > threshold * 2:
        #     print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: motion_latent_len: {motion_latent.shape[0]}, ldmk_len: {ldmk.shape[0]}, appear_latent_len: {appear_latent.shape[0]}, audio_feat_len: {audio_feat.shape[0]}")
        #     return False
        return True
    
    def _check_latent(self,
                      motion_latent,
                      ldmk,
                      appear_latent,
                      audio_feat,
                      pose,
                      file_name,
                      threshold=2):
        if (motion_latent is None) or (ldmk is None) or (appear_latent is None) or (audio_feat is None) or (pose is None):
            return False
        aligned_latent_len = min(int(audio_feat.shape[0] / 2), motion_latent.shape[0], ldmk.shape[0], appear_latent.shape[0], pose.shape[0])
        if aligned_latent_len < self.min_num_frames:
            return False
        if abs(motion_latent.shape[0] - aligned_latent_len) > threshold:
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: motion_latent_len: {motion_latent.shape[0]}, ldmk_len: {ldmk.shape[0]}, appear_latent_len: {appear_latent.shape[0]}, audio_feat_len: {audio_feat.shape[0]}, pose_len: {pose.shape[0]}")
            return False
        if abs(ldmk.shape[0] - aligned_latent_len) > threshold:
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: motion_latent_len: {motion_latent.shape[0]}, ldmk_len: {ldmk.shape[0]}, appear_latent_len: {appear_latent.shape[0]}, audio_feat_len: {audio_feat.shape[0]},pose_len: {pose.shape[0]}")
            return False
        if abs(appear_latent.shape[0] - aligned_latent_len) > threshold:
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: motion_latent_len: {motion_latent.shape[0]}, ldmk_len: {ldmk.shape[0]}, appear_latent_len: {appear_latent.shape[0]}, audio_feat_len: {audio_feat.shape[0]},pose_len: {pose.shape[0]}")
            return False
        if abs(audio_feat.shape[0] - aligned_latent_len * 2) > threshold * 2:
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: motion_latent_len: {motion_latent.shape[0]}, ldmk_len: {ldmk.shape[0]}, appear_latent_len: {appear_latent.shape[0]}, audio_feat_len: {audio_feat.shape[0]},pose_len: {pose.shape[0]}")
            return False
        if abs(pose.shape[0] - aligned_latent_len) > threshold:
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: motion_latent_len: {motion_latent.shape[0]}, ldmk_len: {ldmk.shape[0]}, appear_latent_len: {appear_latent.shape[0]}, audio_feat_len: {audio_feat.shape[0]},pose_len: {pose.shape[0]}")
            return False
        return True

    def _check_file_name(self, file_name):
        test_list = ['M003', 'M007', 'M019', 'W018', 'W038']
        
        blacklist_codec = ['CCv2_part_12_0818_portuguese_nonscripted_1', 'CCv2_part_14_0916_portuguese_nonscripted_1', 'CCv2_part_15_1006_portuguese_nonscripted_1', \
                     'CCv2_part_16_1088_portuguese_nonscripted_1', 'CCv2_part_16_1119_portuguese_nonscripted_1', 'CCv2_part_20_1379_portuguese_nonscripted_1', \
                     'CCv2_part_22_1500_portuguese_nonscripted_1', 'CCv2_part_24_1647_portuguese_nonscripted_1', 'CCv2_part_26_1781_hindi_nonscripted_0', \
                     'CCv2_part_2_0122_portuguese_nonscripted_1', 'CCv2_part_36_2498_hindi_nonscripted_0', 'CCv2_part_36_2498_hindi_nonscripted_1', \
                     'CCv2_part_48_3348_hindi_nonscripted_0', 'CCv2_part_48_3348_hindi_nonscripted_1', 'CCv2_part_51_3498_indonesian_nonscripted_1', \
                     'CCv2_part_51_3533_indonesian_nonscripted_1', 'CCv2_part_54_3713_indonesian_nonscripted_1', 'CCv2_part_54_3731_indonesian_nonscripted_1', \
                     'CCv2_part_54_3740_indonesian_nonscripted_1', 'CCv2_part_54_3746_indonesian_nonscripted_1', 'CCv2_part_57_3924_spanish_nonscripted_1', \
                     'CCv2_part_60_4176_spanish_nonscripted_1', 'CCv2_part_63_4378_tagalog_nonscripted_0', 'CCv2_part_64_4457_tagalog_nonscripted_1', \
                     'CCv2_part_66_4565_tagalog_nonscripted_1', 'CCv2_part_69_4789_tagalog_nonscripted_1', 'CCv2_part_6_0376_portuguese_nonscripted_1', \
                     'CCv2_part_6_0381_portuguese_nonscripted_1', 'CCv2_part_70_4816_tagalog_nonscripted_0', 'CCv2_part_71_4882_tagalog_nonscripted_1', \
                     'CCv2_part_72_4979_tagalog_nonscripted_0', 'CCv2_part_72_4989_tagalog_nonscripted_1', 'CCv2_part_73_5034_tagalog_nonscripted_0', \
                     'CCv2_part_74_5108_tagalog_nonscripted_1', 'CCv2_part_75_5172_tagalog_nonscripted_0', 'CCv2_part_76_5284_english_nonscripted_1', \
                     'CCv2_part_77_5330_english_nonscripted_1', 'CCv2_part_79_5454_english_nonscripted_1', 'CCv2_part_80_5513_english_nonscripted_1', \
                     'CCv2_part_80_5549_english_nonscripted_1', 'CCv2_part_9_0594_portuguese_nonscripted_1']
        blacklist_runyi = ['CCv2_part_12_0824_portuguese_nonscripted_1_0', 'CCv2_part_13_0887_portuguese_scripted_0_0', 'CCv2_part_14_0954_portuguese_nonscripted_1_0', \
                           'CCv2_part_15_0989_portuguese_nonscripted_1_0', 'CCv2_part_22_1470_portuguese_nonscripted_1_0', 'CCv2_part_2_0094_portuguese_scripted_0_2', \
                            'CCv2_part_4_0264_portuguese_scripted_0_1', 'CCv2_part_55_3799_english_nonscripted_4_0', 'CCv2_part_9_0585_portuguese_nonscripted_4_0']
        if file_name in blacklist_codec:
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: blacklist_codec")
            return False
        if file_name in blacklist_runyi:
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: blacklist_runyi")
            return False
        if file_name.split('_')[0] in test_list or int(file_name.split('_')[-2])==1:
            return False
        return True

    def _check_emo_inst(self, inst):
        if not inst:
            return False
        instruction_list = ['angry', 'contempt', 'disgusted', 'fear', 'sad', 'neutral', 'happy', 'surprised']
        if inst in instruction_list:
            return True
        else:
            return False
        
    def _check_au(self, au, file_name):
        if len(au.keys()) != 1:
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: no au or many aus")
            return False
        extracted_file_name = list(au.keys())[0]
        au_list = au[extracted_file_name]
        if '_'.join(extracted_file_name.split('/')) != file_name + '_':
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: au file name not match")
            return False
        if len(au_list) == 0:
            print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: no au")
            return False
        return True
        
        
    def _load_hdtf(self,
                   hdtf_latent_dir_name,
                   hdtf_audio_feat_dir_name,
                   hdtf_pose_dir_name):
        """
        load data path from self.data_root and save to self.data
        """
        print(f"[TalkingA2VqvaeSeqKeylaBase][HDTF] Loading HDTF data from {self.data_root}")
        total_latent_len = 0
        # load latent for each video clip
        latent_files = glob.glob(os.path.join(self.data_root, hdtf_latent_dir_name, 'motion*.npy'))
        import random
        random.shuffle(latent_files)
        # for each video clip
        for latent_file in tqdm(latent_files[:int(0.7*len(latent_files))]):
            # get file name
            file_name = os.path.basename(latent_file).split('.')[0].strip('motion_')
            # check if audio feature exists
            if os.path.isfile(os.path.join(self.data_root, hdtf_audio_feat_dir_name, file_name + '.npy')):
                try:
                    motion_latent = np.load(os.path.join(self.data_root, hdtf_latent_dir_name, 'motion_' + file_name + '.npy'))
                    ldmk = np.load(os.path.join(self.data_root, hdtf_latent_dir_name, 'ldmk_' + file_name + '.npy'))
                    appear_latent = np.load(os.path.join(self.data_root, hdtf_latent_dir_name, 'appear_' + file_name + '.npy'))
                    audio_feat = np.load(os.path.join(self.data_root, hdtf_audio_feat_dir_name, file_name + '.npy')).T  # (2frames, 512)
                    pose = np.load(os.path.join(self.data_root, hdtf_pose_dir_name, file_name + '.npy'))
                except:
                    print(f"[TalkingA2VqvaeSeqKeylaBase][HDTF] Dropping {file_name}")
                    motion_latent = None
                    ldmk = None
                    appear_latent = None
                    audio_feat = None
                    pose = None
                if self._check_latent(motion_latent, ldmk, appear_latent, audio_feat, pose, file_name):
                    single_video_data = dict()
                    single_video_data['filename'] = file_name
                    single_video_data['motion_latent_path'] = os.path.join(self.data_root, hdtf_latent_dir_name, 'motion_' + file_name + '.npy')
                    single_video_data['ldmk_path'] = os.path.join(self.data_root, hdtf_latent_dir_name, 'ldmk_' + file_name + '.npy')
                    single_video_data['appear_latent_path'] = os.path.join(self.data_root, hdtf_latent_dir_name, 'appear_' + file_name + '.npy')
                    single_video_data['audio_feat_path'] = os.path.join(self.data_root, hdtf_audio_feat_dir_name, file_name + '.npy')
                    # add pose path
                    single_video_data['pose_path'] = os.path.join(self.data_root, hdtf_pose_dir_name, file_name + '.npy')
                    single_video_data['instruction'] = '[EMPTY]'
                    single_video_data['audio_flag'] = 1
                    single_video_data['instruction_flag'] = 0
                    single_video_data['action_units'] = '[EMPTY]'
                    total_latent_len += motion_latent.shape[0]
                    self.data.append(single_video_data)
        total_minutes = total_latent_len / 25. / 60.
        print(f"[TalkingA2VqvaeSeqKeylaBase][HDTF] Loaded {len(self.data)} sequences ({total_minutes:.2f} minutes, {total_minutes / 60.:.2f} hours) from {self.data_root}")
      
    def _load_mead(self,
                  ccd_latent_dir_name,
                  ccdv1_audio_feat_dir_name,
                  mead_pose_dir_name,
                  action_units_dir_name):
        """
        load data path from self.data_root and save to self.data
        """
        print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Loading MEAD data from {self.data_root}")
        total_latent_len = 0
        # total_id = []
        # load latent for each video clip
        latent_files = sorted(glob.glob(os.path.join(self.data_root, ccd_latent_dir_name, 'motion*.npy')))  # ALL motion files
        ccd_audio_feat_files = dict()
        ccdv1_audio_feat_filenames = sorted(glob.glob(os.path.join(self.data_root, ccdv1_audio_feat_dir_name, '*.npy')))
        for f in ccdv1_audio_feat_filenames:
            # audio_feat_file_name = '_'.join(os.path.basename(f).split('.')[0].split('_')[4:])
            audio_feat_file_name = os.path.basename(f).split('.')[0]
            if audio_feat_file_name in ccd_audio_feat_files:
                print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Duplicate audio feature file name {audio_feat_file_name}")
            ccd_audio_feat_files[audio_feat_file_name] = f
        
        # for each video clip
        for latent_file in tqdm(latent_files[:int(0.95*len(latent_files))]):
            # get file name
            file_name = os.path.basename(latent_file).split('.')[0].strip('motion_')
            # check if audio feature exists
            if file_name in ccd_audio_feat_files:
                try:
                    motion_latent = np.load(os.path.join(self.data_root, ccd_latent_dir_name, 'motion_' + file_name + '.npy')) # shape (frames, 3, 16, 16)
                    ldmk = np.load(os.path.join(self.data_root, ccd_latent_dir_name, 'ldmk_' + file_name + '.npy'))
                    appear_latent = np.load(os.path.join(self.data_root, ccd_latent_dir_name, 'appear_' + file_name + '.npy'))
                    audio_feat = np.load(ccd_audio_feat_files[file_name]).T  # (2frames, 512)
                    instruction = file_name.split('_')[1]
                    pose = np.load(os.path.join(self.data_root, mead_pose_dir_name, file_name + '.npy'))
                   
                except:
                    print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: reading error")
                    motion_latent = None
                    ldmk = None
                    appear_latent = None
                    audio_feat = None
                    instruction = None
                    pose = None
                    action_units = None
                if self._check_latent(motion_latent, ldmk, appear_latent, audio_feat, pose, file_name) and self._check_file_name(file_name) and self._check_emo_inst(instruction):
                # if self._check_file_name(file_name) and self._check_emo_inst(instruction):
                    single_video_data = dict()
                    single_video_data['filename'] = file_name
                    single_video_data['motion_latent_path'] = os.path.join(self.data_root, ccd_latent_dir_name, 'motion_' + file_name + '.npy')
                    single_video_data['ldmk_path'] = os.path.join(self.data_root, ccd_latent_dir_name, 'ldmk_' + file_name + '.npy')
                    single_video_data['appear_latent_path'] = os.path.join(self.data_root, ccd_latent_dir_name, 'appear_' + file_name + '.npy')
                    single_video_data['audio_feat_path'] = ccd_audio_feat_files[file_name]
                    single_video_data['instruction'] = instruction
                    single_video_data['pose_path'] = os.path.join(self.data_root, mead_pose_dir_name, file_name + '.npy')
                    single_video_data['audio_flag'] = 1
                    single_video_data['instruction_flag'] = 1
                    try:
                        action_units = json.load(open(os.path.join(self.data_root, action_units_dir_name, file_name.split('_')[0], "_".join(file_name.split('_')[1:]), 'intersection.json')))
                        if self._check_au(action_units, file_name):
                            single_video_data['action_units'] = ','.join(list(action_units.values())[0])
                        else:
                            single_video_data['action_units'] = '[EMPTY]'
                    except:
                        single_video_data['action_units'] = '[EMPTY]'
                    # if 'CC_' in file_name:
                    #     id_num = '_'.join(file_name.split('_')[:5])
                    # elif 'CCv2_' in file_name:
                    #     id_num = '_'.join(file_name.split('_')[:4])
                    # else:
                    #     print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Unknown file name {file_name}")
                    # total_id.append(id_num)
                    # total_latent_len += motion_latent.shape[0]
                    self.data.append(single_video_data)
            else:
                print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: no audio feature file")
        # total_id_num = len(set(total_id))
        # print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Loaded ID: {total_id_num} from {self.data_root}")
        total_minutes = total_latent_len / 25. / 60.
        print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Loaded {len(self.data)} sequences ({total_minutes:.2f} minutes, {total_minutes / 60.:.2f} hours) from {self.data_root}")

    def _load_t2m(self, t2m_latent_dir_name):
        print(f"[TalkingA2VqvaeSeqKeylaBase][t2m] Loading Text2motion data from {self.data_root}")
        total_latent_len = 0
        total_id = []
        # load latent for each video clip
        latent_files = sorted(glob.glob(os.path.join(self.data_root, t2m_latent_dir_name, 'motion*.npy')))
        v2m_dict = json.load(open('video2instruction.json', 'r'))
      
        for latent_file in tqdm(latent_files[int(0.95*len(latent_files)):]):
            # get file name
            file_name = os.path.basename(latent_file).split('.')[0].strip('motion_')
            if file_name not in v2m_dict:
                continue
            try:
                instruction = v2m_dict[file_name]
            # check if audio feature exists
            # if file_name in ccd_audio_feat_files:
            
                motion_latent = np.load(os.path.join(self.data_root, t2m_latent_dir_name, 'motion_' + file_name + '.npy'))
                ldmk = np.load(os.path.join(self.data_root, t2m_latent_dir_name, 'ldmk_' + file_name + '.npy'))
                appear_latent = np.load(os.path.join(self.data_root, t2m_latent_dir_name, 'appear_' + file_name + '.npy'))
                # audio_feat = np.load(ccd_audio_feat_files[file_name]).T
            except:
                print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: reading error")
                motion_latent = None
                appear_latent = None
            if self._check_t2m_latent(motion_latent,  appear_latent, ldmk,  file_name) and self._check_file_name(file_name) and self._check_instruction(instruction):
                single_video_data = dict()
                single_video_data['filename'] = file_name
                single_video_data['motion_latent_path'] = os.path.join(self.data_root, t2m_latent_dir_name, 'motion_' + file_name + '.npy')
                single_video_data['ldmk_path'] = os.path.join(self.data_root, t2m_latent_dir_name, 'ldmk_' + file_name + '.npy')
                single_video_data['appear_latent_path'] = os.path.join(self.data_root, t2m_latent_dir_name, 'appear_' + file_name + '.npy')
                single_video_data['audio_feat_path'] = '[EMPTY]'
                single_video_data['instruction'] = instruction
                single_video_data['audio_flag'] = 0
                single_video_data['pose_path'] = '[EMPTY]'
                single_video_data['instruction_flag'] = 1
                # token_embeddings = self.clip_text_embedder.encode(instruction)
                # single_video_data['token_embedding'] = token_embeddings
                if 'CC_' in file_name:
                    id_num = '_'.join(file_name.split('_')[:5])
                elif 'CCv2_' in file_name:
                    id_num = '_'.join(file_name.split('_')[:4])
                else:
                    print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Unknown file name {file_name}")
                total_id.append(id_num)
                total_latent_len += motion_latent.shape[0]
                self.data.append(single_video_data)
            # else:
            #     print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Dropping {file_name}: no audio feature file")
        total_id_num = len(set(total_id))
        print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Loaded ID: {total_id_num} from {self.data_root}")
        total_minutes = total_latent_len / 25. / 60.
        print(f"[TalkingA2VqvaeSeqKeylaBase][CCD] Loaded {len(self.data)} sequences ({total_minutes:.2f} minutes, {total_minutes / 60.:.2f} hours) from {self.data_root}")
    
    def __getitem__(self, i):
        try:
            motion_latent = np.load(self.data[i]['motion_latent_path'])
            ldmk = np.load(self.data[i]['ldmk_path'])
            appear_latent = np.load(self.data[i]['appear_latent_path'])
            if self.data[i]['audio_flag'] == 1:
                audio_feat = np.load(self.data[i]['audio_feat_path']).T
                pose = np.load(self.data[i]['pose_path'])
            else:
                audio_feat = np.zeros((2 * motion_latent.shape[0], self.audio_feat_dim))
                pose = np.zeros((motion_latent.shape[0], self.target_pose_dim))
            inst_flag = self.data[i]['instruction_flag']
            audio_flag = self.data[i]['audio_flag']

            aligned_latent_len = min(int(audio_feat.shape[0] / 2), motion_latent.shape[0], ldmk.shape[0], appear_latent.shape[0], pose.shape[0])
            if aligned_latent_len < self.min_num_frames:
                print(aligned_latent_len, inst_flag, audio_flag)
                return self.__getitem__(i + 1)
            # motion_latent: (T, 3, 16, 16)
            motion_latent = motion_latent[:aligned_latent_len].reshape(aligned_latent_len, self.target_motion_dim)
            # ldmk: (T, 669, 2)
            ldmk = ldmk[:aligned_latent_len].reshape(aligned_latent_len, self.target_ldmk_dim)
            # appear_latent: (T, 3, 64, 64)
            appear_latent = appear_latent[:aligned_latent_len].reshape(aligned_latent_len, self.target_appear_dim)
            # audio_feat: (2*T, 512)
            audio_feat = audio_feat[:2 * aligned_latent_len].reshape(2 * aligned_latent_len, self.audio_feat_dim)
            pose = pose[:aligned_latent_len].reshape(aligned_latent_len, self.target_pose_dim)
            instruction = self.data[i]['instruction']
            action_unit = self.data[i]['action_units']
        except:
            print(f"[TalkingA2VqvaeSeqKeylaBase] Error loading {self.data[i]['filename']}")
            return {
                "motion_latent": self.last_motion_latent,
                "ldmk": self.last_ldmk,
                "appear_latent": self.last_appear_latent,
                "audio_feat": self.last_audio_feat,
                "latent_length": self.last_aligned_latent_len,
                "filename": self.last_filename,
                "pose": self.last_pose,
                "instruction": self.instruction,
                "inst_flag": self.inst_flag,
                "audio_flag": self.audio_flag,
                "action_unit": self.action_unit
            }

        self.last_aligned_latent_len, self.last_motion_latent, self.last_ldmk, self.last_appear_latent, self.last_audio_feat, self.last_filename, self.last_pose, self.inst_flag, self.audio_flag = aligned_latent_len, motion_latent, ldmk, appear_latent, audio_feat, self.data[i]['filename'], pose, inst_flag, audio_flag, 

        return {
            "motion_latent": motion_latent,
            "ldmk": ldmk,
            "appear_latent": appear_latent,
            "audio_feat": audio_feat,
            "latent_length": aligned_latent_len,
            "filename": self.data[i]['filename'],
            "pose": pose,
            "instruction": instruction,
            "inst_flag": inst_flag,
            "audio_flag": audio_flag,
            "action_unit": action_unit
        }

    
        
        
        
def create_lmdb_dataset(data_set: Dataset,
                        save_dir: str,
                        name: str,
                        num_workers=16,
                        max_size_rate=1.0,
                        write_frequency=5000):
    """
    from torchtoolbox.tools.convert_lmdb import generate_lmdb_dataset, raw_reader
    from torchvision.datasets import ImageFolder

    dt = ImageFolder(..., loader=raw_reader)
    save_dir = xxx
    dataset_name = yyy
    generate_lmdb_dataset(dt, save_dir=save_dir, name=dataset_name)
    """
    data_loader = DataLoader(data_set, num_workers=num_workers, collate_fn=lambda x: x)
    meta_info = {'name': name}
    keys = list()
    num_samples = len(data_set)
    check_dir(save_dir)

    lmdb_path = os.path.join(save_dir, '{}.lmdb'.format(name))
    db = lmdb.open(lmdb_path,
                   subdir=False,
                   map_size=int(1099511627776 * max_size_rate),
                   readonly=False,
                   meminit=True,
                   map_async=True)
    txn = db.begin(write=True)

    for idx, data in enumerate(tqdm(data_loader)):
        # token_embeddings = data_set.clip_text_embedder.encode(data[0]['instruction'])
        # data[0]['token_embedding'] = token_embeddings.cpu().numpy()
        keys.append(data[0]['filename'])  # or resolution, etc.
        # data is a list, len = 1
        txn.put(data[0]['filename'].encode('ascii'), dumps_pyarrow(data[0]))
        if idx % write_frequency == 0 and idx > 0:
            txn.commit()
            txn = db.begin(write=True)
    txn.put(b'__len__', dumps_pyarrow(num_samples))

    txn.commit()
    db.sync()
    db.close()
    meta_info['keys'] = keys
    pickle.dump(meta_info, open(os.path.join(save_dir, 'meta_info.pkl'), "wb"))


if __name__ == "__main__":

    data_root = '/xxxx/TETF/datasets/'

    mead_audio_feat_dir_name = 'MEAD_all/audio_wave2vec_all/'
    mead_latent_dir_name = 'MEAD_all/MEAD_all_latents_700M_ep12/'
    mead_pose_dir_name = 'MEAD_all/pose_3ddfa_all/'
    mead_action_units_dir_name = 'MEAD_all/au_detect/'
    
    hdtf_latent_dir_name = 'HDTF/latents-700m-avena-randft-ep12-hdtf/'
    hdtf_audio_feat_dir_name = 'HDTF/audios_16k_wav2vec'
    hdtf_pose_dir_name = 'HDTF/pose_dir'
    
    t2m_latent_dir_name = 'text2motion/text2motion_latents_700M_ep12'
    
    save_dir = '/xxxx/TETF/datasets/lmdbs_au_pose/hdtf_lmdb_all_au_pose_train_subset'
    dataset_name = 'hdtf_lmdb_all_au_pose_train_subset'

    dataset = TalkingA2VqvaeSeqKeylaBase(data_root=data_root,
                                         mead_latent_dir_name=mead_latent_dir_name,
                                         mead_audio_feat_dir_name=mead_audio_feat_dir_name,
                                         mead_pose_dir_name=mead_pose_dir_name,
                                         mead_action_units_dir_name=mead_action_units_dir_name,
                                        #  ccdv2_audio_feat_dir_name=ccdv2_audio_feat_dir_name,
                                         hdtf_latent_dir_name=hdtf_latent_dir_name,
                                         hdtf_audio_feat_dir_name=hdtf_audio_feat_dir_name,
                                         hdtf_pose_dir_name=hdtf_pose_dir_name,
                                         t2m_latent_dir_name=t2m_latent_dir_name,
                                         min_num_frames=25)

    create_lmdb_dataset(dataset, save_dir=save_dir, name=dataset_name)

