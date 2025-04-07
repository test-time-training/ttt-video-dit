#!/bin/bash

# This script is for launching training on multiple nodes

# Update these values
CODE_PATH="YOUR_CODE_PATH"
DATE="TODO"
DUMP_FOLDER="$CODE_PATH/experiments/3sec"

cd ${CODE_PATH} || exit 1

conda activate ttt-video

export TRITON_CACHE_DIR='/tmp/triton_cache'
export OMP_NUM_THREADS=1

# Add this
export WANDB_API_KEY='TODO'

NUM_NODES=8
TIMEOUT_MINUTES=240

SLURM_ACCOUNT="TODO"
SLURM_PARTITION="TODO"

# Update with your intended starting weights
CHECKPOINT_WEIGHTS_DIR=''
CONFIG_FILE="./configs/train/ttt-mlp/3s.toml"

EXP_NAME="${DATE}-ttt-video-3s-BS64-5000steps"

overrides="--nodes=${NUM_NODES} \
            --partition=${SLURM_PARTITION} \
            --account=${SLURM_ACCOUNT} \
            --timeout=${TIMEOUT_MINUTES} \
            --checkpoint.timeout_minutes=${TIMEOUT_MINUTES} \
            --job.config_file ${CONFIG_FILE} \
            --job.exp_name=${EXP_NAME} \
            --job.dump_folder=${DUMP_FOLDER} \
            --checkpoint.init_state_dir=${CHECKPOINT_WEIGHTS_DIR} \
            --checkpoint.resume \
            --checkpoint.resume_step=-1 \
        "

job_id=$(python train_submitit.py ${overrides} | grep "Submitted job_id:" | awk '{print $3}')
echo "Job id: ${job_id}"
