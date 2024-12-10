#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, glob, random, time
import lmdb
import pickle
import pyarrow
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn


def load_pyarrow(buf):
    assert buf is not None, 'buf should not be None.'
    return pyarrow.deserialize(buf)


class TalkingA2VqvaeSeqKeylaSepeLMDBBase(Dataset):
    def __init__(self,
                 data_root,
                 lmdb_dir_list,
                 lmdb_name_list=None,
                 target_key='motion_latent',  # 'motion_latent' or 'ldmk'
                 keyla_key='motion_latent',  # 'motion_latent' or 'appear_latent' or 'ldmk'
                 meta_info_path='meta_info.pkl',
                 data_root2=None,
                 data_root3=None,
                 min_num_frames=70,
                 max_num_frames=250,
                 target_motion_dim=768,
                 target_ldmk_dim=1338,
                 target_appear_dim=12288,
                 norm_mean=None,
                 norm_std=None,
                 audio_feat_dim=512,
                 pre_load_lmdb=False):
        """Dataset for HDTF dataset
        NOTE: must filter the data with min_num_frames, recommended to give some buffer, e.g. min_num_frames=70, but filtered with 75
        """
        self.data_root = self._verify_paths(data_root, data_root2, data_root3)
        self.lmdb_dir_list = lmdb_dir_list
        self.lmdb_name_list = lmdb_name_list
        if lmdb_name_list is not None:
            assert len(lmdb_dir_list) == len(lmdb_name_list), f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] len(lmdb_dir_list) {len(lmdb_dir_list)} should be equal to len(lmdb_name_list) {len(lmdb_name_list)}"
        assert min_num_frames < max_num_frames, f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] min_num_frames {min_num_frames} should be smaller than max_num_frames {max_num_frames}"
        assert target_key in ['motion_latent', 'ldmk'], f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] target_key {target_key} should be either motion_latent or ldmk"
        assert keyla_key in ['motion_latent', 'appear_latent', 'ldmk'], f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] keyla_key {keyla_key} should be either motion_latent or appear_latent"

        rank_zero_info(f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] Assume that latents is filtered by min_num_frames {min_num_frames}")
        self.target_key = target_key
        self.keyla_key = keyla_key
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames
        self.target_latent_dim = target_motion_dim if target_key == 'motion_latent' else target_ldmk_dim
        if keyla_key == 'motion_latent':
            self.target_keyla_dim = target_motion_dim
        elif keyla_key == 'appear_latent':
            self.target_keyla_dim = target_appear_dim
        elif keyla_key == 'ldmk':
            self.target_keyla_dim = target_ldmk_dim
        else:
            raise NotImplementedError()
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.audio_feat_dim = audio_feat_dim
        self.pre_load_lmdb = pre_load_lmdb

        self.keys = list()
        self.envs = list()
        self.txns = list()
        self.keys_envs_mapping = dict()
        self.keys_txns_mapping = dict()
        self.opened_lmdb_dir_list = list()
        self.opened_lmdb_name_list = list()
        # prepare and load the data, prepare for reading in getitem
        self._prepare()
        self._load(lmdb_dir_list, lmdb_name_list, meta_info_path)

        assert len(self.keys) > 0, f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] No data found in {self.data_root}"

    def __len__(self):
        return len(self.keys)

    def _verify_paths(self, path1, path2, path3):
        if (path1 is not None) and os.path.isdir(path1):
            return path1
        elif (path2 is not None) and os.path.isdir(path2):
            return path2
        elif (path3 is not None) and os.path.isdir(path3):
            return path3
        else:
            rank_zero_info(f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] Invalid dirs {path1}, {path2}, {path3}")

    def _prepare(self):
        raise NotImplementedError()

    def _normalize_latent(self, latent):
        if (self.norm_mean is not None) and (self.norm_std is not None):
            latent = (latent - self.norm_mean) / self.norm_std
        return latent

    def _load_single_lmdb(self, lmdb_dir, lmdb_name, meta_info_path):
        meta_info = pickle.load(open(os.path.join(self.data_root, lmdb_dir, meta_info_path), "rb"))
        # update keys
        self.keys.extend(meta_info['keys'])
        if self.pre_load_lmdb:
            # open lmdb
            env = lmdb.open(os.path.join(self.data_root, lmdb_dir, '{}.lmdb'.format(lmdb_name)),
                            subdir=False,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            # check length
            # with env.begin() as txn:
            #     data_len = load_pyarrow(txn.get(b'__len__'))
            # assert data_len == len(meta_info['keys']), f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] data_len {data_len} should be equal to len(meta_info['keys']) {len(meta_info['keys'])}"
            # update keys_envs_mapping and envs
            rank_zero_info(f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] Loaded {lmdb_dir} with {len(meta_info['keys'])} data")
            self.envs.append(env)
            self.keys_envs_mapping.update({key: len(self.envs) - 1 for key in meta_info['keys']})
        else:
            self.envs.append(None)
            self.txns.append(None)
            self.keys_txns_mapping.update({key: len(self.txns) - 1 for key in meta_info['keys']})
            self.opened_lmdb_dir_list.append(lmdb_dir)
            self.opened_lmdb_name_list.append(lmdb_name)

    def _load_single_lmdbdir(self, lmdb_dir, lmdb_name, meta_info_path):
        if lmdb_name is None:
            lmdb_name = glob.glob(os.path.join(self.data_root, lmdb_dir, '*.lmdb'))[0].split('/')[-1].split('.')[0]
        if os.path.isfile(os.path.join(self.data_root, lmdb_dir, '{}.lmdb'.format(lmdb_name))):
            self._load_single_lmdb(lmdb_dir, lmdb_name, meta_info_path)
        else:
            rank_zero_info(f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] Invalid ldmb dir {lmdb_dir} & lmdb name {lmdb_name}")

    def _load(self, lmdb_dir_list, lmdb_name_list, meta_info_path):
        """
        load lmdb from self.data_root, keys are stored in self.keys
        """
        rank_zero_info(f"[TETFTalkingA2VqvaeSeqKeylaSepeLMDBBase] Loading data from {self.data_root}")

        lmdb_name_list = [None] * len(lmdb_dir_list) if lmdb_name_list is None else lmdb_name_list
        for lmdb_dir, lmdb_name in zip(lmdb_dir_list, lmdb_name_list):
            if os.path.isfile(os.path.join(self.data_root, lmdb_dir, meta_info_path)):
                self._load_single_lmdbdir(lmdb_dir, lmdb_name, meta_info_path)
            else:
                for sub_dir in os.listdir(os.path.join(self.data_root, lmdb_dir)):
                    if os.path.isdir(os.path.join(self.data_root, lmdb_dir, sub_dir)) and \
                        os.path.isfile(os.path.join(self.data_root, lmdb_dir, sub_dir, meta_info_path)):
                        self._load_single_lmdbdir(os.path.join(lmdb_dir, sub_dir), lmdb_name, meta_info_path)
                    else:
                        rank_zero_info(f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] Invalid ldmb dir {lmdb_dir} & sub dir {sub_dir} & lmdb name {lmdb_name}")

    def _get_segment_index(self, latent_length):
        raise NotImplementedError()

    def _open_lmdb(self, lmdb_index):
        self.envs[lmdb_index] = lmdb.open(os.path.join(self.data_root, self.opened_lmdb_dir_list[lmdb_index], '{}.lmdb'.format(self.opened_lmdb_name_list[lmdb_index])),
                                          subdir=False,
                                          readonly=True,
                                          lock=False,
                                          readahead=False,
                                          meminit=False)
        self.txns[lmdb_index] = self.envs[lmdb_index].begin()

    def __getitem__(self, i):
        """
        load latent and audio_feat for a single video clip
        align the lengths of latent and audio_feat
        """
        if self.pre_load_lmdb:
            with self.envs[self.keys_envs_mapping[self.keys[i]]].begin() as txn:
                data = load_pyarrow(txn.get(self.keys[i].encode('ascii')))
        else:
            if self.txns[self.keys_txns_mapping[self.keys[i]]] is None:
                self._open_lmdb(self.keys_txns_mapping[self.keys[i]])
            data = load_pyarrow(self.txns[self.keys_txns_mapping[self.keys[i]]].get(self.keys[i].encode('ascii')))

        # np.frombuffer
        latent = data[self.target_key]
        latent = latent.reshape(latent.shape[0], self.target_latent_dim)
        latent = self._normalize_latent(latent)
        audio_feat = data['audio_feat']
        latent_length = data['latent_length']
        latent_filename = data['filename']
        headpose = data['headpose']
        assert latent_length >= self.min_num_frames, f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] file {latent_filename}: latent_length {latent_length} should be larger than min_num_frames {self.min_num_frames}"

        ### randomly select keylatent and repeat
        keylatent = data[self.keyla_key]
        # to fix appear latent mismatch dim
        keylatent = keylatent.reshape(keylatent.shape[0], self.target_keyla_dim)[random.randrange(0, latent_length)]
        keylatent = np.repeat(keylatent[np.newaxis, :], latent_length, axis=0)

        start_index, end_index = self._get_segment_index(latent_length)

        # init latent, audio_feat, keylatent
        latent_paded = torch.zeros((self.max_num_frames, self.target_latent_dim), dtype=torch.float32)
        audio_feat_paded = torch.zeros((self.max_num_frames * 2, self.audio_feat_dim), dtype=torch.float32)
        keylatent_paded = torch.zeros((self.max_num_frames, self.target_keyla_dim), dtype=torch.float32)
        headpose_paded = torch.zeros((self.max_num_frames, 3), dtype=torch.float32)
        # copy data
        latent_paded[:end_index - start_index] = torch.from_numpy(latent[start_index:end_index])
        audio_feat_paded[:2 * (end_index - start_index)] = torch.from_numpy(audio_feat[2 * start_index:2 * end_index])
        keylatent_paded[:end_index - start_index] = torch.from_numpy(keylatent[start_index:end_index])
        headpose_paded[:end_index - start_index] = torch.from_numpy(headpose[start_index:end_index])

        return {
            "latent": latent_paded,
            "latent_length": end_index - start_index,
            "audio_feat": audio_feat_paded,
            "audio_feat_length": 2 * (end_index - start_index),
            "keylatent": keylatent_paded,
            "headpose": headpose_paded,
            "latent_filename": latent_filename
        }


class TalkingA2VqvaeSeqKeylaSepeLMDBTrain(TalkingA2VqvaeSeqKeylaSepeLMDBBase):
    def _prepare(self):
        rank_zero_info(f"[TalkingA2VqvaeSeqKeylaSepeLMDBTrain] Loading training data")

    def _get_segment_index(self, latent_length):
        # randomly select a start index
        start_index = random.randrange(latent_length - self.min_num_frames + 1)
        end_index = min(latent_length, start_index + random.randrange(self.min_num_frames, self.max_num_frames + 1))
        return start_index, end_index


class TalkingA2VqvaeSeqKeylaSepeLMDBValidation(TalkingA2VqvaeSeqKeylaSepeLMDBBase):
    def _prepare(self):
        rank_zero_info(f"[TalkingA2VqvaeSeqKeylaSepeLMDBValidation] Loading validation data")

    def _get_segment_index(self, latent_length):
        # randomly select a start index
        start_index = 0
        end_index = min(self.max_num_frames, latent_length)
        return start_index, end_index
    
    
    
class TETFPoseTalkingA2VqvaeSeqKeyla(TalkingA2VqvaeSeqKeylaSepeLMDBBase):
    def __init__(self,
                 data_root,
                 lmdb_dir_list,
                 lmdb_name_list=None,
                 target_key='motion_latent',  # 'motion_latent' or 'ldmk'
                 keyla_key='motion_latent',  # 'motion_latent' or 'appear_latent' or 'ldmk'
                 meta_info_path='meta_info.pkl',
                 data_root2=None,
                 data_root3=None,
                 min_num_frames=70,
                 max_num_frames=250,
                 target_motion_dim=768,
                 target_ldmk_dim=1338,
                 target_appear_dim=12288,
                 instruction_key='instruction',
                 au_key='action_unit',
                 norm_mean=None,
                 norm_std=None,
                 audio_feat_dim=512,
                 pre_load_lmdb=False,
                 use_au=False):
        
        super().__init__(data_root,
                         lmdb_dir_list,
                         lmdb_name_list,
                         target_key,
                         keyla_key,
                         meta_info_path,
                         data_root2,
                         data_root3,
                         min_num_frames,
                         max_num_frames,
                         target_motion_dim,
                         target_ldmk_dim,
                         target_appear_dim,
                         norm_mean,
                         norm_std,
                         audio_feat_dim,
                         pre_load_lmdb)
        
        self.instruction_key = instruction_key
        self.au_key = au_key
        self.use_au = use_au
                    
    
    def __getitem__(self, i):
        """
        load latent and audio_feat for a single video clip
        align the lengths of latent and audio_feat
        """
        if self.pre_load_lmdb:
            with self.envs[self.keys_envs_mapping[self.keys[i]]].begin() as txn:
                data = load_pyarrow(txn.get(self.keys[i].encode('ascii')))
        else:
            if self.txns[self.keys_txns_mapping[self.keys[i]]] is None:
                self._open_lmdb(self.keys_txns_mapping[self.keys[i]])
            data = load_pyarrow(self.txns[self.keys_txns_mapping[self.keys[i]]].get(self.keys[i].encode('ascii')))

        # np.frombuffer
        latent = data[self.target_key]
        latent = latent.reshape(latent.shape[0], self.target_latent_dim)
        latent = self._normalize_latent(latent)
        audio_feat = data['audio_feat']
        latent_length = data['latent_length']
        latent_filename = data['filename']
        headpose = data['pose']
        instruction = data[self.instruction_key]
        assert latent_length >= self.min_num_frames, f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] file {latent_filename}: latent_length {latent_length} should be larger than min_num_frames {self.min_num_frames}"

        ### randomly select keylatent and repeat
        keylatent = data[self.keyla_key]
        # to fix appear latent mismatch dim
        keylatent = keylatent.reshape(keylatent.shape[0], self.target_keyla_dim)[random.randrange(0, latent_length)]  # randomly select a keylatent
        keylatent = np.repeat(keylatent[np.newaxis, :], latent_length, axis=0)

        start_index, end_index = self._get_segment_index(latent_length)
        
        # init latent, audio_feat, keylatent
        latent_paded = torch.zeros((self.max_num_frames, self.target_latent_dim), dtype=torch.float32)
        audio_feat_paded = torch.zeros((self.max_num_frames * 2, self.audio_feat_dim), dtype=torch.float32)
        keylatent_paded = torch.zeros((self.max_num_frames, self.target_keyla_dim), dtype=torch.float32)
        headpose_paded = torch.zeros((self.max_num_frames, 3), dtype=torch.float32)
        # copy data
        latent_paded[:end_index - start_index] = torch.from_numpy(latent[start_index:end_index])
        audio_feat_paded[:2 * (end_index - start_index)] = torch.from_numpy(audio_feat[2 * start_index:2 * end_index])
        keylatent_paded[:end_index - start_index] = torch.from_numpy(keylatent[start_index:end_index])
        headpose_paded[:end_index - start_index] = torch.from_numpy(headpose[start_index:end_index])
        
        return {
                "latent": latent_paded,
                "latent_length": end_index - start_index,
                "audio_feat": audio_feat_paded,
                "audio_feat_length": 2 * (end_index - start_index),
                "keylatent": keylatent_paded,
                "headpose": headpose_paded,
                "latent_filename": latent_filename,
                "instructs": instruction
            }                

class TETFPoseTalkingA2VqvaeSeqKeylaSepeLMDBTrain(TETFPoseTalkingA2VqvaeSeqKeyla):
    def _prepare(self):
        rank_zero_info(f"[TETFTalkingA2VqvaeSeqKeylaSepeLMDBTrain] Loading training data")

    def _get_segment_index(self, latent_length):
        # randomly select a start index
        start_index = random.randrange(latent_length - self.min_num_frames + 1)
        end_index = min(latent_length, start_index + random.randrange(self.min_num_frames, self.max_num_frames + 1))
        return start_index, end_index


class TETFPoseTalkingA2VqvaeSeqKeylaSepeLMDBValidation(TETFPoseTalkingA2VqvaeSeqKeyla):
    
    def _prepare(self):
        rank_zero_info(f"[TETFTalkingA2VqvaeSeqKeylaSepeLMDBValidation] Loading validation data")

    def _get_segment_index(self, latent_length):
        # randomly select a start index
        start_index = 0
        end_index = min(self.max_num_frames, latent_length)
        return start_index, end_index
    
    

class TETFAllTalkingA2VqvaeSeqKeyla(TETFPoseTalkingA2VqvaeSeqKeyla):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __getitem__(self, i):
        """
        load latent and audio_feat for a single video clip
        align the lengths of latent and audio_feat
        """
        if self.pre_load_lmdb:
            with self.envs[self.keys_envs_mapping[self.keys[i]]].begin() as txn:
                data = load_pyarrow(txn.get(self.keys[i].encode('ascii')))
        else:
            if self.txns[self.keys_txns_mapping[self.keys[i]]] is None:
                self._open_lmdb(self.keys_txns_mapping[self.keys[i]])
            data = load_pyarrow(self.txns[self.keys_txns_mapping[self.keys[i]]].get(self.keys[i].encode('ascii')))

        # np.frombuffer
        latent = data[self.target_key] # (num_frames, 768)
        latent = latent.reshape(latent.shape[0], self.target_latent_dim)
        latent = self._normalize_latent(latent)
        audio_feat = data['audio_feat'] # (num_frames * 2, 512)
        latent_length = data['latent_length']
        if latent_length < self.min_num_frames - 1:
            return self.__getitem__(i + 1)
        latent_filename = data['filename']
        test_person_list = ['M003', "M019", 'W038', "W018", "M007" ]
        if latent_filename.split('_')[0] in test_person_list:
            return self.__getitem__(i + 1)
        headpose = data['pose'] # (num_frames, 3)
        instruction = data[self.instruction_key] 
        inst_flag = data['inst_flag']
        audio_flag = data['audio_flag']
        # assert latent_length >= self.min_num_frames, f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] file {latent_filename}: latent_length {latent_length} should be larger than min_num_frames {self.min_num_frames}"

        ### randomly select keylatent and repeat
        keylatent = data[self.keyla_key] # (num_frames, 768)
        # to fix appear latent mismatch dim
        keylatent = keylatent.reshape(keylatent.shape[0], self.target_keyla_dim)[random.randrange(0, latent_length)]  # randomly select a keylatent
        keylatent = np.repeat(keylatent[np.newaxis, :], latent_length, axis=0)

        start_index, end_index = self._get_segment_index(latent_length)
        
        # init latent, audio_feat, keylatent
        latent_paded = torch.zeros((self.max_num_frames, self.target_latent_dim), dtype=torch.float32)
        audio_feat_paded = torch.zeros((self.max_num_frames * 2, self.audio_feat_dim), dtype=torch.float32)
        keylatent_paded = torch.zeros((self.max_num_frames, self.target_keyla_dim), dtype=torch.float32)
        headpose_paded = torch.zeros((self.max_num_frames, 3), dtype=torch.float32)
        # copy data
        latent_paded[:end_index - start_index] = torch.from_numpy(latent[start_index:end_index])
        audio_feat_paded[:2 * (end_index - start_index)] = torch.from_numpy(audio_feat[2 * start_index:2 * end_index])
        keylatent_paded[:end_index - start_index] = torch.from_numpy(keylatent[start_index:end_index])
        headpose_paded[:end_index - start_index] = torch.from_numpy(headpose[start_index:end_index])
        if self.use_au:
            action_units = data[self.au_key]
            return {
                "latent": latent_paded,
                "latent_length": end_index - start_index,
                "audio_feat": audio_feat_paded,
                "audio_feat_length": 2 * (end_index - start_index),
                "keylatent": keylatent_paded,
                "headpose": headpose_paded,
                "latent_filename": latent_filename,
                "instructs": instruction,
                "inst_flag": inst_flag,
                "audio_flag": audio_flag,
                "action_units": action_units
            }            
        else:
            return {
                "latent": latent_paded,
                "latent_length": end_index - start_index,
                "audio_feat": audio_feat_paded,
                "audio_feat_length": 2 * (end_index - start_index),
                "keylatent": keylatent_paded,
                "headpose": headpose_paded,
                "latent_filename": latent_filename,
                "instructs": instruction,
                "inst_flag": inst_flag,
                "audio_flag": audio_flag
            }                
       
    
class TETFAllTalkingA2VqvaeSeqKeylaSepeLMDBTrain(TETFAllTalkingA2VqvaeSeqKeyla):
    def _prepare(self):
        rank_zero_info(f"[TETFAllTalkingA2VqvaeSeqKeylaSepeLMDBTrain] Loading training data")

    def _get_segment_index(self, latent_length):
        # randomly select a start index
        if latent_length - self.min_num_frames + 1 <= 0:
            print(latent_length)
            start_index = 0
        else:
            start_index = random.randrange(latent_length - self.min_num_frames + 1)
        end_index = min(latent_length, start_index + random.randrange(self.min_num_frames, self.max_num_frames + 1))
        return start_index, end_index


class TETFAllTalkingA2VqvaeSeqKeylaSepeLMDBValidation(TETFAllTalkingA2VqvaeSeqKeyla):
    
    def _prepare(self):
        rank_zero_info(f"[TETFAllTalkingA2VqvaeSeqKeylaSepeLMDBValidation] Loading validation data")

    def _get_segment_index(self, latent_length):
        # max_num_frames or true length
        start_index = 0
        end_index = min(self.max_num_frames, latent_length)
        return start_index, end_index
    
    
    
class TETFAllRandomKeylaTalkingA2VqvaeSeqKeyla(TETFAllTalkingA2VqvaeSeqKeyla):
    def __init__(self, use_random_emotion=False, use_neutral=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_neutral = use_neutral
        self.use_random_emotion = use_random_emotion
    def __getitem__(self, i):
        """
        load latent and audio_feat for a single video clip
        align the lengths of latent and audio_feat
        
        key latent is randomly selected from the other emotion/ neutral
        """
        
        current_key = self.keys[i]
        current_id = current_key.split('_')[0]
        current_id_list = [i for i in self.keys if i.split('_')[0] == current_id]
        try:
            if self.use_random_emotion:
                new_key = random.choice(current_id_list)
            elif self.use_neutral:
                new_key = random.choice([i for i in current_id_list if i.split('_')[1] == 'neutral'])
            else:
                print('error!')
            
            if self.pre_load_lmdb:
                with self.envs[self.keys_envs_mapping[self.keys[i]]].begin() as txn:
                    data = load_pyarrow(txn.get(self.keys[i].encode('ascii')))
                    key_latent_data = load_pyarrow(txn.get(new_key.encode('ascii')))
            else:
                if self.txns[self.keys_txns_mapping[self.keys[i]]] is None:
                    self._open_lmdb(self.keys_txns_mapping[self.keys[i]])
                if self.txns[self.keys_txns_mapping[new_key]] is None:
                    self._open_lmdb(self.keys_txns_mapping[new_key])
                data = load_pyarrow(self.txns[self.keys_txns_mapping[self.keys[i]]].get(self.keys[i].encode('ascii')))
                key_latent_data = load_pyarrow(self.txns[self.keys_txns_mapping[new_key]].get(new_key.encode('ascii')))
        except:
            print('use original data')
            key_latent_data = data
        # np.frombuffer
        latent = data[self.target_key] # (num_frames, 768)
        latent = latent.reshape(latent.shape[0], self.target_latent_dim)
        latent = self._normalize_latent(latent)
        audio_feat = data['audio_feat'] # (num_frames * 2, 512)
        latent_length = data['latent_length']
        if latent_length < self.min_num_frames - 1:
            return self.__getitem__(i + 1)
        latent_filename = data['filename']
        test_person_list = ['M003', "M019", 'W038', "W018", "M007" ]
        if latent_filename.split('_')[0] in test_person_list:
            return self.__getitem__(i + 1)
        headpose = data['pose'] # (num_frames, 3)
        instruction = data[self.instruction_key] 
        inst_flag = data['inst_flag']
        audio_flag = data['audio_flag']
        # assert latent_length >= self.min_num_frames, f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] file {latent_filename}: latent_length {latent_length} should be larger than min_num_frames {self.min_num_frames}"

        ### randomly select keylatent and repeat
        keylatent = key_latent_data[self.keyla_key] # (num_frames, 768)
        # to fix appear latent mismatch dim
        keylatent = keylatent.reshape(keylatent.shape[0], self.target_keyla_dim)[random.randrange(0, key_latent_data['latent_length'])]  # randomly select a keylatent
        keylatent = np.repeat(keylatent[np.newaxis, :], latent_length, axis=0)

        start_index, end_index = self._get_segment_index(latent_length)
        
        # init latent, audio_feat, keylatent
        latent_paded = torch.zeros((self.max_num_frames, self.target_latent_dim), dtype=torch.float32)
        audio_feat_paded = torch.zeros((self.max_num_frames * 2, self.audio_feat_dim), dtype=torch.float32)
        keylatent_paded = torch.zeros((self.max_num_frames, self.target_keyla_dim), dtype=torch.float32)
        headpose_paded = torch.zeros((self.max_num_frames, 3), dtype=torch.float32)
        # copy data
        latent_paded[:end_index - start_index] = torch.from_numpy(latent[start_index:end_index])
        audio_feat_paded[:2 * (end_index - start_index)] = torch.from_numpy(audio_feat[2 * start_index:2 * end_index])
        keylatent_paded[:end_index - start_index] = torch.from_numpy(keylatent[start_index:end_index])
        headpose_paded[:end_index - start_index] = torch.from_numpy(headpose[start_index:end_index])
        if self.use_au:
            action_units = data[self.au_key]
            return {
                "latent": latent_paded,
                "latent_length": end_index - start_index,
                "audio_feat": audio_feat_paded,
                "audio_feat_length": 2 * (end_index - start_index),
                "keylatent": keylatent_paded,
                "headpose": headpose_paded,
                "latent_filename": latent_filename,
                "instructs": instruction,
                "inst_flag": inst_flag,
                "audio_flag": audio_flag,
                "action_units": action_units
            }            
        else:
            return {
                "latent": latent_paded,
                "latent_length": end_index - start_index,
                "audio_feat": audio_feat_paded,
                "audio_feat_length": 2 * (end_index - start_index),
                "keylatent": keylatent_paded,
                "headpose": headpose_paded,
                "latent_filename": latent_filename,
                "instructs": instruction,
                "inst_flag": inst_flag,
                "audio_flag": audio_flag
            }
            
class TETFAllRandomKeylaTalkingA2VqvaeSeqKeylaSepeLMDBTrain(TETFAllRandomKeylaTalkingA2VqvaeSeqKeyla):
    def _prepare(self):
        rank_zero_info(f"[TETFAllRandomKeylaTalkingA2VqvaeSeqKeylaSepeLMDBTrain] Loading training data")

    def _get_segment_index(self, latent_length):
        # randomly select a start index
        if latent_length - self.min_num_frames + 1 <= 0:
            print(latent_length)
            start_index = 0
        else:
            start_index = random.randrange(latent_length - self.min_num_frames + 1)
        end_index = min(latent_length, start_index + random.randrange(self.min_num_frames, self.max_num_frames + 1))
        return start_index, end_index
    
class TETFAllRandomKeylaTalkingA2VqvaeSeqKeylaSepeLMDBValidation(TETFAllRandomKeylaTalkingA2VqvaeSeqKeyla):
    
    def _prepare(self):
        rank_zero_info(f"[TETFAllRandomKeylaTalkingA2VqvaeSeqKeylaSepeLMDBValidation] Loading validation data")

    def _get_segment_index(self, latent_length):
        # max_num_frames or true length
        start_index = 0
        end_index = min(self.max_num_frames, latent_length)
        return start_index, end_index                
    
    
class TETFUnifyRandomKeylaTalkingA2VqvaeSeqKeyla(TETFAllTalkingA2VqvaeSeqKeyla):
    def __init__(self, use_random_emotion=False, use_neutral=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_neutral = use_neutral
        self.use_random_emotion = use_random_emotion
    def __getitem__(self, i):
        """
        load latent and audio_feat for a single video clip
        align the lengths of latent and audio_feat
        
        key latent is randomly selected from the other emotion/ neutral
        """
        
        # current_key = self.keys[i]
        # current_id = current_key.split('_')[0]
        # current_id_list = [i for i in self.keys if i.split('_')[0] == current_id]
        # try:
        #     if self.use_random_emotion:
        #         new_key = random.choice(current_id_list)
        #     elif self.use_neutral:
        #         new_key = random.choice([i for i in current_id_list if i.split('_')[1] == 'neutral'])
        #     else:
        #         print('error!')
            
        if self.pre_load_lmdb:
            with self.envs[self.keys_envs_mapping[self.keys[i]]].begin() as txn:
                data = load_pyarrow(txn.get(self.keys[i].encode('ascii')))
                # key_latent_data = load_pyarrow(txn.get(new_key.encode('ascii')))
        else:
            if self.txns[self.keys_txns_mapping[self.keys[i]]] is None:
                self._open_lmdb(self.keys_txns_mapping[self.keys[i]])
            # if self.txns[self.keys_txns_mapping[new_key]] is None:
            #     self._open_lmdb(self.keys_txns_mapping[new_key])
            data = load_pyarrow(self.txns[self.keys_txns_mapping[self.keys[i]]].get(self.keys[i].encode('ascii')))
            # key_latent_data = load_pyarrow(self.txns[self.keys_txns_mapping[new_key]].get(new_key.encode('ascii')))
        # except:
        #     print('use original data')
        #     key_latent_data = data
        # np.frombuffer
        inst_flag = data['inst_flag']
        audio_flag = data['audio_flag']
        latent = data[self.target_key] # (num_frames, 768)
        latent = latent.reshape(latent.shape[0], self.target_latent_dim)
        latent = self._normalize_latent(latent)
        audio_feat = data['audio_feat'] # (num_frames * 2, 512)
        latent_length = data['latent_length']
        if latent_length < self.min_num_frames - 1:
            return self.__getitem__(i + 1)
        latent_filename = data['filename']
        test_person_list = ['M003', "M019", 'W038', "W018", "M007" ]
        if latent_filename.split('_')[0] in test_person_list:
            return self.__getitem__(i + 1)
        headpose = data['pose'] # (num_frames, 3)
        instruction = data[self.instruction_key] 
        
        # assert latent_length >= self.min_num_frames, f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] file {latent_filename}: latent_length {latent_length} should be larger than min_num_frames {self.min_num_frames}"

        if inst_flag and audio_flag and (self.use_neutral or self.use_random_emotion): # namely MEAD data
            current_key = self.keys[i]
            current_id = current_key.split('_')[0]
            current_id_list = [i for i in self.keys if i.split('_')[0] == current_id]
            try:
                if self.use_random_emotion:
                    new_key = random.choice(current_id_list)
                elif self.use_neutral:
                    new_key = random.choice([i for i in current_id_list if i.split('_')[1] == 'neutral'])
                if self.pre_load_lmdb:
                    with self.envs[self.keys_envs_mapping[self.keys[i]]].begin() as txn:
                        # data = load_pyarrow(txn.get(self.keys[i].encode('ascii')))
                        key_latent_data = load_pyarrow(txn.get(new_key.encode('ascii')))
                else:
                    # if self.txns[self.keys_txns_mapping[self.keys[i]]] is None:
                    #     self._open_lmdb(self.keys_txns_mapping[self.keys[i]])
                    if self.txns[self.keys_txns_mapping[new_key]] is None:
                        self._open_lmdb(self.keys_txns_mapping[new_key])
                    key_latent_data = load_pyarrow(self.txns[self.keys_txns_mapping[new_key]].get(new_key.encode('ascii')))
                    # data = load_pyarrow(self.txns[self.keys_txns_mapping[self.keys[i]]].get(self.keys[i].encode('ascii')))
            except:
                key_latent_data = data
        else:
            key_latent_data = data
        ### randomly select keylatent and repeat
        keylatent = key_latent_data[self.keyla_key] # (num_frames, 768)
        # to fix appear latent mismatch dim
        keylatent = keylatent.reshape(keylatent.shape[0], self.target_keyla_dim)[random.randrange(0, key_latent_data['latent_length'])]  # randomly select a keylatent
        keylatent = np.repeat(keylatent[np.newaxis, :], latent_length, axis=0)

        start_index, end_index = self._get_segment_index(latent_length)
        
        # init latent, audio_feat, keylatent
        latent_paded = torch.zeros((self.max_num_frames, self.target_latent_dim), dtype=torch.float32)
        audio_feat_paded = torch.zeros((self.max_num_frames * 2, self.audio_feat_dim), dtype=torch.float32)
        keylatent_paded = torch.zeros((self.max_num_frames, self.target_keyla_dim), dtype=torch.float32)
        headpose_paded = torch.zeros((self.max_num_frames, 3), dtype=torch.float32)
        # copy data
        latent_paded[:end_index - start_index] = torch.from_numpy(latent[start_index:end_index])
        audio_feat_paded[:2 * (end_index - start_index)] = torch.from_numpy(audio_feat[2 * start_index:2 * end_index])
        keylatent_paded[:end_index - start_index] = torch.from_numpy(keylatent[start_index:end_index])
        headpose_paded[:end_index - start_index] = torch.from_numpy(headpose[start_index:end_index])
        if self.use_au:
            action_units = data[self.au_key]
            return {
                "latent": latent_paded,
                "latent_length": end_index - start_index,
                "audio_feat": audio_feat_paded,
                "audio_feat_length": 2 * (end_index - start_index),
                "keylatent": keylatent_paded,
                "headpose": headpose_paded,
                "latent_filename": latent_filename,
                "instructs": instruction,
                "inst_flag": inst_flag,
                "audio_flag": audio_flag,
                "action_units": action_units
            }            
        else:
            return {
                "latent": latent_paded,
                "latent_length": end_index - start_index,
                "audio_feat": audio_feat_paded,
                "audio_feat_length": 2 * (end_index - start_index),
                "keylatent": keylatent_paded,
                "headpose": headpose_paded,
                "latent_filename": latent_filename,
                "instructs": instruction,
                "inst_flag": inst_flag,
                "audio_flag": audio_flag
            }
            
            
class TETFUnifyRandomKeylaTalkingA2VqvaeSeqKeylaSepeLMDBTrain(TETFUnifyRandomKeylaTalkingA2VqvaeSeqKeyla):
    def _prepare(self):
        rank_zero_info(f"[TETFUnifyRandomKeylaTalkingA2VqvaeSeqKeylaSepeLMDBTrain] Loading training data")

    def _get_segment_index(self, latent_length):
        # randomly select a start index
        if latent_length - self.min_num_frames + 1 <= 0:
            print(latent_length)
            start_index = 0
        else:
            start_index = random.randrange(latent_length - self.min_num_frames + 1)
        end_index = min(latent_length, start_index + random.randrange(self.min_num_frames, self.max_num_frames + 1))
        return start_index, end_index
    
    
class TETFUnifyRandomKeylaTalkingA2VqvaeSeqKeylaSepeLMDBValidation(TETFUnifyRandomKeylaTalkingA2VqvaeSeqKeyla):
    
    def _prepare(self):
        rank_zero_info(f"[TETFUnifyRandomKeylaTalkingA2VqvaeSeqKeylaSepeLMDBValidation] Loading validation data")

    def _get_segment_index(self, latent_length):
        # max_num_frames or true length
        start_index = 0
        end_index = min(self.max_num_frames, latent_length)
        return start_index, end_index
    
    
class VarLenTETFUnifyRandomKeylaTalkingA2VqvaeSeqKeyla(TETFAllTalkingA2VqvaeSeqKeyla):
    def __init__(self, use_random_emotion=False, use_neutral=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_neutral = use_neutral
        self.use_random_emotion = use_random_emotion
    def __getitem__(self, i):
        """
        load latent and audio_feat for a single video clip
        align the lengths of latent and audio_feat
        
        key latent is randomly selected from the other emotion/ neutral
        """
        
            
        if self.pre_load_lmdb:
            with self.envs[self.keys_envs_mapping[self.keys[i]]].begin() as txn:
                data = load_pyarrow(txn.get(self.keys[i].encode('ascii')))
                # key_latent_data = load_pyarrow(txn.get(new_key.encode('ascii')))
        else:
            if self.txns[self.keys_txns_mapping[self.keys[i]]] is None:
                self._open_lmdb(self.keys_txns_mapping[self.keys[i]])
            # if self.txns[self.keys_txns_mapping[new_key]] is None:
            #     self._open_lmdb(self.keys_txns_mapping[new_key])
            data = load_pyarrow(self.txns[self.keys_txns_mapping[self.keys[i]]].get(self.keys[i].encode('ascii')))
            # key_latent_data = load_pyarrow(self.txns[self.keys_txns_mapping[new_key]].get(new_key.encode('ascii')))
        # except:
        #     print('use original data')
        #     key_latent_data = data
        # np.frombuffer
        inst_flag = data['inst_flag']
        audio_flag = data['audio_flag']
        latent = data[self.target_key] # (num_frames, 768)
        latent = latent.reshape(latent.shape[0], self.target_latent_dim)
        latent = self._normalize_latent(latent)
        audio_feat = data['audio_feat'] # (num_frames * 2, 512)
        latent_length = data['latent_length']
        if latent_length < self.min_num_frames - 1:
            return self.__getitem__(i + 1)
        latent_filename = data['filename']
        test_person_list = ['M003', "M019", 'W038', "W018", "M007" ]
        if latent_filename.split('_')[0] in test_person_list:
            return self.__getitem__(i + 1)
        headpose = data['pose'] # (num_frames, 3)
        instruction = data[self.instruction_key] 
        
        # assert latent_length >= self.min_num_frames, f"[TalkingA2VqvaeSeqKeylaSepeLMDBBase] file {latent_filename}: latent_length {latent_length} should be larger than min_num_frames {self.min_num_frames}"

        if inst_flag and audio_flag and (self.use_neutral or self.use_random_emotion): # namely MEAD data
            current_key = self.keys[i]
            current_id = current_key.split('_')[0]
            current_id_list = [i for i in self.keys if i.split('_')[0] == current_id]
            try:
                if self.use_random_emotion:
                    new_key = random.choice(current_id_list)
                elif self.use_neutral:
                    new_key = random.choice([i for i in current_id_list if i.split('_')[1] == 'neutral'])
                if self.pre_load_lmdb:
                    with self.envs[self.keys_envs_mapping[self.keys[i]]].begin() as txn:
                        # data = load_pyarrow(txn.get(self.keys[i].encode('ascii')))
                        key_latent_data = load_pyarrow(txn.get(new_key.encode('ascii')))
                else:
                    # if self.txns[self.keys_txns_mapping[self.keys[i]]] is None:
                    #     self._open_lmdb(self.keys_txns_mapping[self.keys[i]])
                    if self.txns[self.keys_txns_mapping[new_key]] is None:
                        self._open_lmdb(self.keys_txns_mapping[new_key])
                    key_latent_data = load_pyarrow(self.txns[self.keys_txns_mapping[new_key]].get(new_key.encode('ascii')))
                    # data = load_pyarrow(self.txns[self.keys_txns_mapping[self.keys[i]]].get(self.keys[i].encode('ascii')))
            except:
                key_latent_data = data
        else:
            key_latent_data = data
        ### randomly select keylatent and repeat
        keylatent = key_latent_data[self.keyla_key] # (num_frames, 768)
        # to fix appear latent mismatch dim
        keylatent = keylatent.reshape(keylatent.shape[0], self.target_keyla_dim)[random.randrange(0, key_latent_data['latent_length'])]  # randomly select a keylatent
        keylatent = np.repeat(keylatent[np.newaxis, :], latent_length, axis=0)
        if not audio_flag:
            start_index, end_index = 0, latent_length
        else:
            start_index, end_index = self._get_segment_index(latent_length)
        
        # # init latent, audio_feat, keylatent
        # latent_paded = torch.zeros((self.max_num_frames, self.target_latent_dim), dtype=torch.float32)
        # audio_feat_paded = torch.zeros((self.max_num_frames * 2, self.audio_feat_dim), dtype=torch.float32)
        # keylatent_paded = torch.zeros((self.max_num_frames, self.target_keyla_dim), dtype=torch.float32)
        # headpose_paded = torch.zeros((self.max_num_frames, 3), dtype=torch.float32)
        # # copy data
        # latent_paded[:end_index - start_index] = torch.from_numpy(latent[start_index:end_index])
        # audio_feat_paded[:2 * (end_index - start_index)] = torch.from_numpy(audio_feat[2 * start_index:2 * end_index])
        # keylatent_paded[:end_index - start_index] = torch.from_numpy(keylatent[start_index:end_index])
        # headpose_paded[:end_index - start_index] = torch.from_numpy(headpose[start_index:end_index])
        latent_paded = torch.from_numpy(latent[start_index:end_index])
        audio_feat_paded = torch.from_numpy(audio_feat[2 * start_index:2 * end_index])
        keylatent_paded = torch.from_numpy(keylatent[start_index:end_index])
        headpose_paded = torch.from_numpy(headpose[start_index:end_index])
        if self.use_au:
            action_units = data[self.au_key]
            return {
                "latent": latent_paded,
                "latent_length": end_index - start_index,
                "audio_feat": audio_feat_paded,
                "audio_feat_length": 2 * (end_index - start_index),
                "keylatent": keylatent_paded,
                "headpose": headpose_paded,
                "latent_filename": latent_filename,
                "instructs": instruction,
                "inst_flag": inst_flag,
                "audio_flag": audio_flag,
                "action_units": action_units
            }            
        else:
            return {
                "latent": latent_paded,
                "latent_length": end_index - start_index,
                "audio_feat": audio_feat_paded,
                "audio_feat_length": 2 * (end_index - start_index),
                "keylatent": keylatent_paded,
                "headpose": headpose_paded,
                "latent_filename": latent_filename,
                "instructs": instruction,
                "inst_flag": inst_flag,
                "audio_flag": audio_flag
            }

class VarLenTETFUnifyRandomKeylaTalkingA2VqvaeSeqKeylaSepeLMDBTrain(VarLenTETFUnifyRandomKeylaTalkingA2VqvaeSeqKeyla):
    def _prepare(self):
        rank_zero_info(f"[VarLenTETFUnifyRandomKeylaTalkingA2VqvaeSeqKeylaSepeLMDBTrain] Loading training data")

    def _get_segment_index(self, latent_length):
        # randomly select a start index
        if latent_length - self.min_num_frames + 1 <= 0:
            print(latent_length)
            start_index = 0
        else:
            start_index = random.randrange(latent_length - self.min_num_frames + 1)
        end_index = min(latent_length, start_index + random.randrange(self.min_num_frames, self.max_num_frames + 1))
        return start_index, end_index
    
class VarLenTETFUnifyRandomKeylaTalkingA2VqvaeSeqKeylaSepeLMDBValidation(VarLenTETFUnifyRandomKeylaTalkingA2VqvaeSeqKeyla):
        
        def _prepare(self):
            rank_zero_info(f"[VarLenTETFUnifyRandomKeylaTalkingA2VqvaeSeqKeylaSepeLMDBValidation] Loading validation data")
    
        def _get_segment_index(self, latent_length):
            # max_num_frames or true length
            start_index = 0
            end_index = min(self.max_num_frames, latent_length)
            return start_index, end_index
        
        
class VarLenTETFUnifyRandomKeylaTalkingCollateFunc:
    def __init__(self, target_dim, audio_feat_dim, pose_dim) -> None:
        self.target_dim = target_dim
        self.audio_dim = audio_feat_dim
        self.pose_dim = pose_dim
        
    def pad_sequence(self, seqs, max_seq_length, target_dim):
        res = []
        max_seq_length = min(max_seq_length, 250)
        for seq in seqs:
            seq_paded = torch.zeros((max_seq_length, target_dim), dtype=torch.float32)
            seq_paded[:min(seq.shape[0], max_seq_length)] = seq[:min(seq.shape[0], max_seq_length)]
            res.append(seq_paded)
        return torch.stack(res)
    
    def pad_audio_sequence(self, seqs, max_seq_length, target_dim):
        res = []
        max_seq_length = min(max_seq_length, 500)
        for seq in seqs:
            seq_paded = torch.zeros((max_seq_length, target_dim), dtype=torch.float32)
            seq_paded[:min(seq.shape[0], max_seq_length)] = seq[:min(seq.shape[0], max_seq_length)]
            res.append(seq_paded)
        return torch.stack(res)
    
    def __call__(self, batch_dic):
        batch_len = len(batch_dic) # batch size
        max_seq_length = max([dic['latent_length'] for dic in batch_dic]) # the longest sequence length in a batch
        # mask_batch=torch.zeros((batch_len, max_seq_length)) # mask
        latent_batch=[]
        keylatent_batch=[]
        inst_batch=[]
        latent_filename_batch = []
        latent_len_batch = []
        audio_feat_batch = []
        audio_feat_len_batch = []
        headpose_batch = []
        inst_flag_batch = []
        audio_flag_batch = []
        action_units_batch = []
        for i in range(len(batch_dic)): # extract data from batch_dic
            dic = batch_dic[i]
            latent_batch.append(dic['latent'])
            keylatent_batch.append(dic['keylatent'])
            latent_filename_batch.append(dic['latent_filename'])
            latent_len_batch.append(torch.tensor(dic['latent_length']))
            inst_batch.append(dic['instructs'])
            audio_feat_batch.append(dic['audio_feat'])
            audio_feat_len_batch.append(dic['audio_feat_length'])
            headpose_batch.append(dic['headpose'])
            inst_flag_batch.append(torch.tensor(dic['inst_flag']))
            audio_flag_batch.append(torch.tensor(dic['audio_flag']))
            if 'action_units' in dic:
                action_units_batch.append(dic['action_units'])
            
            # id_batch.append(dic['id'])
            # mask_batch[i,:dic['length']]=1 # mask
            
        res={}
        
        res['latent'] = self.pad_sequence(latent_batch, max_seq_length, self.target_dim) # 将信息封装在字典res中
        res['keylatent'] = self.pad_sequence(keylatent_batch, max_seq_length, self.target_dim)
        res['instructs'] = inst_batch
        res['latent_filename'] = latent_filename_batch
        res['latent_length'] = torch.stack(latent_len_batch)
        res['audio_feat'] = self.pad_audio_sequence(audio_feat_batch, max_seq_length * 2, self.audio_dim)
        res['audio_feat_length'] = audio_feat_len_batch
        res['headpose'] = self.pad_sequence(headpose_batch, max_seq_length, self.pose_dim)
        res['inst_flag'] = torch.stack(inst_flag_batch)
        res['audio_flag'] = torch.stack(audio_flag_batch)
        if len(action_units_batch) > 0:
            res['action_units'] = action_units_batch
        return res