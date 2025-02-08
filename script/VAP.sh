#!/bin/sh

experiment_id=$(uuidgen | cut -c 1-8)
echo "Experiment ID: $experiment_id"

CUDA_VISIBLE_DEVICES=0 python -u code_LVMs/cure.py \
    --dataset pope_adversarial \
    --model internvl2-8b \
    --experiment_id $experiment_id \
    --prompt_origin True \
    --record_words record \
    --dataset_root dataset_path \
    --model_path model_path \
    --clip_model ViT-L/14 \
    --debug False \
    --record_path record_cure \
    --ddpm_t 200 \
    --lambda1 1 \
    --lambda2 0.5 \
    --lambda3 0.5 \
    --epsilon 2 \
    --alpha 1 \
    --steps 1 \
    --num_query 5 \
    --sigma 8 

# LLaVA 1 1 1 False 500 alpha=4
# Instruct-BLIP 1 1 1 False 500
# Qwen2-VL 1 0.1 0.5 False 500
# InternVL2-MPO 1 0.5 0.5 False 800
# LLAVA-OV 0.1 1 0.1 False
# Ovis 1 1 1 False 500
# Deepseek 1 1 1 False 100