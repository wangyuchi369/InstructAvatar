#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

INST_FLAG=True
AUDIO_FLAG=True
# A2E_CONFIG=configs/a2vqvae-hdtf-ccd-avena-pose-ep12/unified/split/unified_with_pose_small_unify_split_varlen.yaml
# CKPT=/mnt/blob/xxxx/TETF/exp_logs/unified/lr1e-5-dp0.08-unified_from_gaia_zero_conv_split_varlen/2024-01-21T18-05-12_unified_with_pose_small_unify_split_varlen/checkpoints/last.ckpt
A2E_CONFIG=$1
CKPT=$2
USE_EOS=${3:-False}
TEST_SET=/mnt/blob/xxxx/TETF/datasets/talkinghead-1kh/
VIDEO_DIR=/mnt/blob/xxxx/TETF/eval_results_unify_outdomain/$(echo $CKPT | cut -d'/' -f 8)_$(echo $CKPT | cut -d'/' -f 11 | cut -d'.' -f 1)/
# VIDEO_DIR=/mnt/blob/xxxx/TETF/eval_results/lr1e-5-dp0.08-finetune_all_params_from_gaia_hdtf_subset_epoch=003799/
python talking/scripts/evaluate_benchmark_outdomain.py \
    --a2e_ckpt $CKPT \
    --a2e_config $A2E_CONFIG \
    --inst_flag $INST_FLAG \
    --audio_flag $AUDIO_FLAG \
    --test_set_dir $TEST_SET \
    --video_save_dir $VIDEO_DIR \
    --use_eos $USE_EOS \
    --use_mead_audio True

python talking/scripts/calculate_au_score_outdomain.py \
    --result_dir $VIDEO_DIR \

python talking/scripts/lip_sync_score_outdomain.py \
    --result_dir $VIDEO_DIR \
   
cat $VIDEO_DIR/final_results.txt


