#!/bin/bash

# Load Conda setup to ensure the "conda activate" command works
source ~/anaconda3/etc/profile.d/conda.sh
mkdir log_temp

experiment_id_global=$(uuidgen | cut -c 1-8)

# Function to run a task with a specific Conda environment
run_task() {
    local conda_env=$1       # Conda environment name
    local experiment_id=$2   # Experiment ID
    local cuda_device=$3     # GPU device
    local model_script=$4    # Path to the script
    local model_path=$5      # Path to the model
    local model_name=$6      # Model name
    local log_file=$7        # Log file to store the output

    # Check if experiment_id is empty, generate a random one if needed
    if [ -z "$experiment_id" ]; then
        experiment_id=$experiment_id_global
        echo "Generated random Experiment ID: $experiment_id"
    fi

    echo "Activating Conda environment: $conda_env"
    conda activate $conda_env

    echo "Experiment ID: $experiment_id"
    CUDA_VISIBLE_DEVICES=$cuda_device python -u $model_script \
        --dataset pope_random \
        --experiment_id $experiment_id \
        --record_words record_words \
        --dataset_root dataset_path \
        --model_path $model_path \
        --debug False \
        --model $model_name \
        --record_path record_path
    conda deactivate
    echo "Task with Experiment ID $experiment_id completed."
}

# Task 1: Instruct-BLIP
run_task internvl "" 0 code_LVMs/vqa.py \
    path_instructblip-vicuna-7b instruct-blip-7b log_temp/instruct-blip.log

# Task 2: InternVL2
run_task internvl "" 0 code_LVMs/vqa.py \
    path_InternVL2-8B internvl2-8b  log_temp/internvl.log

# Task 3: InternVL2-MPO
run_task internvl "" 0 code_LVMs/vqa.py \
    path_InternVL2-8B-MPO internvl2-8b-mpo log_temp/internvl-mpo.log

# Task 4: LLaVA
run_task llava "" 0 code_LVMs/vqa.py \
    path_llava-v1.5-7b llava-v1.5-7b log_temp/llava.log

# Task 5: LLaVA-ov
run_task llava_onevision "" 0 code_LVMs/vqa.py \
    path_llava-onevision-qwen2-7b-ov llava-ov-7b log_temp/llava-ov.log

# Task 6: Ovis1.6-gemma2
run_task llava_onevision "" 0 code_LVMs/vqa.py \
    path_Ovis1.6-Gemma2-9B ovis1.6-gemma2-9b log_temp/ovis1_6.log

# Task 7: Qwen-VL
run_task qwen "" 0 code_LVMs/vqa.py \
    path_Qwen2-VL-7B-Instruct qwen-vl-7b log_temp/qwen-vl.log

# Task 8: DeepSeek-vl2-small
run_task deepseek "" 0 code_LVMs/vqa.py \
    path_deepseek-vl2 deepseek log_temp/deepseek.log
