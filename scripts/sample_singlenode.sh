#!/bin/bash
#SBATCH --job-name=sample-ttt
#SBATCH --account=TODO
#SBATCH --partition=TODO
#SBATCH --nodes=1              # Number of nodes
#SBATCH --gres=gpu:8           # Number of GPUs per node
#SBATCH --ntasks=1             # Number of tasks
#SBATCH --cpus-per-task=8      # Number of CPU cores per task
#SBATCH --time=01:20:00        # Time limit (hh:mm:ss)
#SBATCH --output=slurm/slurm-%j.out  # Standard output and error log

if [ ! -d ".git" ]; then
	echo "Please run this script from the root of the project."
	exit 1
fi

conda activate ttt-video

export WANDB_API_KEY='OPTIONAL'
NUM_GPUS=${NUM_GPUS:-8}

CHECKPOINT_WEIGHTS_DIR="TODO"
CONFIG_FILE="./configs/eval/ttt-mlp/9s.toml"
INPUT_FILE="./inputs/example-9s.json"

torchrun --nproc_per_node="${NUM_GPUS}" \
	--rdzv_backend c10d \
	--rdzv_endpoint="localhost:0" \
	--local-ranks-filter 0 \
	--role rank \
	--tee 3 \
	sample.py \
	--job.config_file ${CONFIG_FILE} \
	--eval.input_file=${INPUT_FILE} \
	--checkpoint.init_state_dir=${CHECKPOINT_WEIGHTS_DIR} \
	--parallelism.dp_replicate="${NUM_GPUS}" \
	--parallelism.dp_sharding=1 \
	--parallelism.tp_sharding=1 \
	--wandb.alert \
	--wandb.disable
