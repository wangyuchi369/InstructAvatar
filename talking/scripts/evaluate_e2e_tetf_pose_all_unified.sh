#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)


APPEAR_DIR=test_cases/demo/cartoon3
# INSTRUCTION="talk with following action units: upper_lip_raiser;lid_tightener;lip_corner_puller;cheek_raiser;lips_part"
INSTRUCTION="be happy"
# AUDIO_DIR=/xxxx/TETF/datasets/MEAD_all/MEAD_all_randcrop/W033/audio_16k/happy_level_3_039_0.wav
AUDIO_DIR=test_cases/hdtf_indomain/audio/Xiaoxiao.wav
# AUDIO_FEAT=/xxxx/TETF/datasets/MEAD_extend/audio_wave2vec_all/M019_happy_002_0.npy
A2E_CONFIG=configs/a2vqvae-hdtf-ccd-avena-pose-ep12/unified/split/unified_with_pose_small_unify_split_varlen_eos.yaml
INST_FLAG=True
AUDIO_FLAG=True
NOTES=audio_feat
path_part=$(echo $INSTRUCTION | sed 's/ /_/g')_inst_${INST_FLAG}_audio_${AUDIO_FLAG}_${NOTES}
CKPT=/xxxx/xxx.ckpt
VIDEO_NAME="text2motion_result/$(echo $CKPT | cut -d'/' -f 7)_$(echo $CKPT | cut -d'/' -f 10 | cut -d'.' -f 1)/$(echo $APPEAR_DIR | rev | cut -d'/' -f 1 | rev)/$path_part.mp4"
# VIDEO_NAME="text2motion_result/unified/inter/$(echo $APPEAR_DIR | rev | cut -d'/' -f 1 | rev)/$path_part.mp4"
python talking/scripts/evaluate_e2e_tetf_pose_all_unified.py \
    --video_name $VIDEO_NAME \
    --appear_dir $APPEAR_DIR \
    --a2e_ckpt $CKPT \
    --audio_path $AUDIO_DIR \
    --instruction "$INSTRUCTION" \
    --a2e_config $A2E_CONFIG \
    --inst_flag $INST_FLAG \
    --audio_flag $AUDIO_FLAG \
    --use_eos True \