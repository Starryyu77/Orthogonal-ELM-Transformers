#!/bin/bash -l
#SBATCH --job-name=oelm-v2-p4
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=120:00:00
#SBATCH --output=logs/oelm-pretrain-v2/phase4_%j.out
#SBATCH --error=logs/oelm-pretrain-v2/phase4_%j.err

set -euo pipefail

METHOD=${1:-baseline}
PROJECT_ROOT=${OELM_PROJECT_ROOT:-/projects/LlamaFactory/OELM-Pretrain}
CONDA_ENV=${OELM_CONDA_ENV:-/projects/LlamaFactory/miniconda3/envs/oelm}

mkdir -p "${PROJECT_ROOT}/logs/oelm-pretrain-v2"
mkdir -p "${PROJECT_ROOT}/outputs/oelm-pretrain-v2"
mkdir -p "${PROJECT_ROOT}/results/oelm-pretrain-v2"

cd "${PROJECT_ROOT}"
module load Miniforge3
source activate "${CONDA_ENV}"

torchrun \
  --nproc_per_node=2 \
  --master_port=29502 \
  experiments/oelm-pretrain-v2/scripts/pretrain_v2.py \
  --phase phase4_openwebtext_formal \
  --dataset openwebtext \
  --method "${METHOD}" \
  --model_size small \
  --output_root "${PROJECT_ROOT}/outputs/oelm-pretrain-v2" \
  --seed 42 \
  --max_steps 100000 \
  --batch_size 32 \
  --gradient_accumulation_steps 2 \
  --seq_length 1024 \
  --learning_rate 3e-4 \
  --warmup_steps 5000 \
  --save_steps 10000 \
  --eval_steps 5000 \
  --logging_steps 100 \
  --validation_split_pct 0.01 \
  --max_eval_batches 100
