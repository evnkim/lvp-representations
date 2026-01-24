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
#SBATCH --mem-per-cpu=4G
#SBATCH --signal=SIGTERM@30
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

echo "My slurm job id is $SLURM_JOB_ID"
echo "My slurm job name is $SLURM_JOB_NAME"
echo "I am running on the following node: $SLURMD_NODENAME"

cd /data/scene-rep/u/evnkim/Projects/large-video-planner

source /data/scene-rep/u/evnkim/.bashrc

# -----------------------------------------------------------------------------
# Clean, job-local environment (avoids any leftover packages/caches in lvp2).
# This is slower (it reinstalls per job) but maximally reproducible.
# -----------------------------------------------------------------------------
MINICONDA="/data/scene-rep/u/evnkim/miniconda3"
CONDA="${MINICONDA}/bin/conda"

ENV_PREFIX="${SLURM_TMPDIR:-/tmp}/lvp2_${SLURM_JOB_ID}"
echo "Using job env prefix: ${ENV_PREFIX}"

rm -rf "${ENV_PREFIX}" || true
rm -rf ~/.cache/pip ~/.cache/torch ~/.cache/flash_attn 2>/dev/null || true

${CONDA} create -y -p "${ENV_PREFIX}" python=3.10
source "${MINICONDA}/bin/activate" "${ENV_PREFIX}"

python -m pip install -U pip setuptools wheel

# Ensure torch/torchvision come from the PyTorch CUDA wheel index.
# (Matches torch==2.6.0+cu124 that this repo expects.)
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124"

python -m pip install --no-cache-dir \
  torch==2.6.0 \
  torchvision==0.21.0

python -m pip install --no-cache-dir -r requirements.txt

# FlashAttention is optional; we try to build it against the *current* torch.
# If it fails, the code should fall back to SDPA.
export CUDA_HOME=/usr/local/cuda
export CUDACXX="${CUDA_HOME}/bin/nvcc"
python -m pip uninstall -y flash-attn || true
python -m pip install --no-cache-dir flash-attn --no-build-isolation || true

# Help the dynamic linker find libtorch for extensions (flash-attn/xformers/etc).
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"


python -m scratch.linear_probe_wan \
  --ckpt-path data/ckpts/Wan2.1-I2V-14B-480P \
  --tuned-ckpt-path data/ckpts/lvp_14B.ckpt \
  --vae-ckpt-path data/ckpts/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \
  --text-ckpt-path data/ckpts/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
  --subset-name imagenet-1k \
  --data-root /data/scene-rep/ImageNet1K \
  --batch-size 4 \
  --num-workers 16 \
  --text-device cpu \
  --t-value 500 \
  --log-every 50 \
  --wandb
