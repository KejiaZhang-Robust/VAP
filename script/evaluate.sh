#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python -u evaluate_hallucination.py \
    --record_root folder_name \
    --dataset pope \
    --experiment_id id \
    --evaluate_file file_name.json \
    --model internvl2-8b \
    --record_path record_path_name
