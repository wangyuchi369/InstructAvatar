export PYTHONPATH=$PYTHONPATH:$(pwd)

folder_path=/mnt/blob/xxx/dataset/CCD/CCDv1_processed/text2express/
ckpt=/mnt/blob/xxxx/diffusion/stable-diffusion/latents/seperate_ldmk_scale_avena_dense_vae_set2_randft_ep12.ckpt
save_path=/mnt/blob/xxxx/TETF/datasets/text2motion/text2motion_latents_700M_ep12/


mkdir -p ${save_path}

python preprocess/infer_latent/infer_latent_text2motion.py --ckpt ${ckpt} --ccd \
    --folder_path ${folder_path} \
    --save_path ${save_path} \
    --config configs/vqvae_2enc-hdtf/tetf_autoencoder_scale_avena_extend_dense_64x64x3.yaml \
    --split 300 --num $1
