#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)



INST_FLAG=True
AUDIO_FLAG=True
# A2E_CONFIG=configs/a2vqvae-hdtf-ccd-avena-pose-ep12/finetune_all_params_mead_au_from_gaia_small.yaml
# CKPT=/mnt/blob/xxxx/TETF/exp_logs/small/lr1e-5-dp0.08-finetune_all_params_from_gaia_hdtf/2024-01-07T00-39-31_finetune_all_params_mead_au_from_gaia_small_with_hdtf/checkpoints/epoch=005999.ckpt
A2E_CONFIG=$1
CKPT=$2
USE_EOS=${3:-False}
TEST_SET=/mnt/blob/xxxx/TETF/datasets/MEAD_extend/
VIDEO_DIR=/mnt/blob/xxxx/TETF/eval_results_unify/$(echo $CKPT | cut -d'/' -f 8)_$(echo $CKPT | cut -d'/' -f 11 | cut -d'.' -f 1)/
# VIDEO_DIR=/mnt/blob/xxxx/TETF/eval_results/lr1e-5-dp0.08-finetune_all_params_from_gaia_hdtf_subset_epoch=003799/
python talking/scripts/evaluate_benchmark.py \
    --a2e_ckpt $CKPT \
    --a2e_config $A2E_CONFIG \
    --inst_flag $INST_FLAG \
    --audio_flag $AUDIO_FLAG \
    --test_set_dir $TEST_SET \
    --video_save_dir $VIDEO_DIR \
    --use_eos $USE_EOS

python talking/scripts/calculate_au_score.py \
    --result_dir $VIDEO_DIR \

python talking/scripts/lip_sync_score.py \
    --result_dir $VIDEO_DIR \
   
cat $VIDEO_DIR/final_results.txt


