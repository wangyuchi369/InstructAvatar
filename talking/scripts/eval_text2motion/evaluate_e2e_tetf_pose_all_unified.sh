#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)


APPEAR_DIR=test_cases/mead_indomain/appear/M019
# INSTRUCTION="talk with following action units: upper_lip_raiser;lid_tightener;lip_corner_puller;cheek_raiser;lips_part"
INSTRUCTION="lower you brower and depress your lip corner"
# AUDIO_DIR=/xxxx/TETF/datasets/MEAD_all/MEAD_all_randcrop/W033/audio_16k/happy_level_3_039_0.wav
AUDIO_DIR=/xxxx/TETF/datasets/MEAD/agg_MEAD/M019/audio_wav/happy_002.wav
# AUDIO_FEAT=/xxxx/TETF/datasets/MEAD_extend/audio_wave2vec_all/M019_happy_002_0.npy
A2E_CONFIG=configs/a2vqvae-hdtf-ccd-avena-pose-ep12/unified/split/unified_with_pose_small_unify_split_varlen_eos_intensity_para_varkey_all_data.yaml
INST_FLAG=True
AUDIO_FLAG=False
NOTES=audio_feat
path_part=$(echo $INSTRUCTION | sed 's/ /_/g')_inst_${INST_FLAG}_audio_${AUDIO_FLAG}_${NOTES}
CKPT=/xxxx/TETF/exp_logs/unified_split_para/lr1e-5-dp0.08-unified_split_eos_intensity_para_varkey_itp/2024-02-21T20-13-38_unified_with_pose_small_unify_split_varlen_eos_intensity_para_varkey_all_data/checkpoints/epoch=003199.ckpt
# VIDEO_NAME="text2motion_result/unified/inter/$(echo $APPEAR_DIR | rev | cut -d'/' -f 1 | rev)/$path_part.mp4"
python talking/scripts/eval_text2motion/evaluate_e2e_tetf_pose_all_unified.py \
    --a2e_ckpt $CKPT \
    --a2e_config $A2E_CONFIG \
    --inst_flag $INST_FLAG \
    --audio_flag $AUDIO_FLAG \
