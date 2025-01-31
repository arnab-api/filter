#!/bin/bash

source ~/.bashrc
# Initialize conda for shell script
eval "$(conda shell.bash hook)"

# activate conda environment
conda activate corner
conda env list

#export your required environment variables below
#################################################
# WANDB_API_KEY=<>

# cd to project directory
cd /disk/u/arnab/Codes/Projects/retrieval

# run the script
# CUDA_VISIBLE_DEVICES=6 python -m scripts.cache_probing_latents --model="meta-llama/Llama-3.2-3B" --probe_class="profession/actors" --limit=10000 --save_dir="probing_latents" |& tee logs/llama-3b/actors.log
