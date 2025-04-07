#!/bin/bash
#SBATCH --job-name=TODO
#SBATCH --account=TODO
#SBATCH --partition=TODO
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:30:00
#SBATCH --output=slurm/slurm-%j.out

torchrun --nproc_per_node=1 dev/precomp_video.py
