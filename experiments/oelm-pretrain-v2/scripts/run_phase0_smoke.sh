#!/bin/bash -l
#SBATCH --job-name=oelm-v2-p0
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/oelm-pretrain-v2/phase0_%j.out
#SBATCH --error=logs/oelm-pretrain-v2/phase0_%j.err

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
  --phase phase0_smoke \
  --dataset tinystories \
  --method "${METHOD}" \
  --model_size mini \
  --output_root "${PROJECT_ROOT}/outputs/oelm-pretrain-v2" \
  --seed 42 \
  --max_steps 200 \
  --batch_size 16 \
  --seq_length 256 \
  --learning_rate 5e-4 \
  --warmup_steps 20 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 20 \
  --max_eval_batches 10
