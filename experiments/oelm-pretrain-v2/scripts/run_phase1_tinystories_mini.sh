#!/bin/bash -l
#SBATCH --job-name=oelm-v2-p1
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/oelm-pretrain-v2/phase1_%j.out
#SBATCH --error=logs/oelm-pretrain-v2/phase1_%j.err

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

python experiments/oelm-pretrain-v2/scripts/pretrain_v2.py \
  --phase phase1_tinystories_mini \
  --dataset tinystories \
  --method "${METHOD}" \
  --model_size mini \
  --output_root "${PROJECT_ROOT}/outputs/oelm-pretrain-v2" \
  --seed 42 \
  --max_steps 10000 \
  --batch_size 32 \
  --seq_length 512 \
  --learning_rate 5e-4 \
  --warmup_steps 1000 \
  --save_steps 2000 \
  --eval_steps 500 \
  --logging_steps 100 \
  --max_eval_batches 50
