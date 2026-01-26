#!/bin/bash
#
#SBATCH --job-name=dec
#SBATCH --partition=vision-shared-h200,csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --account=vision-sitzmann

#SBATCH --time=24:00:00
#SBATCH --output=./big/output/%j.log
#SBATCH --error=./big/error/%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=12G
#SBATCH --signal=SIGTERM@30
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

echo "My slurm job id is $SLURM_JOB_ID"
echo "My slurm job name is $SLURM_JOB_NAME"
echo "I am running on the following node: $SLURMD_NODENAME"

cd /data/scene-rep/u/evnkim/Projects/large-video-planner

source /data/scene-rep/u/evnkim/.bashrc
source /data/scene-rep/u/evnkim/miniconda3/etc/profile.d/conda.sh
conda activate lvp2


python -m scratch.linear_probe_wan \
  --ckpt-path data/ckpts/Wan2.1-I2V-14B-480P \
  --tuned-ckpt-path data/ckpts/LVP_14B_inference.ckpt \
  --vae-ckpt-path data/ckpts/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \
  --text-ckpt-path data/ckpts/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
  --subset-name imagenet-1k \
  --data-root /data/scene-rep/ImageNet1K \
  --batch-size 64 \
  --num-workers 16 \
  --text-device cuda \
  --t-value 500 \
  --log-every 50 \
  --ckpt-dir checkpoints/linear_probe_1k \
  --wandb
