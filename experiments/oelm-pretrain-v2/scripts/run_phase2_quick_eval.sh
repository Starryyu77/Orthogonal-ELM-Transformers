#!/bin/bash -l
#SBATCH --job-name=oelm-v2-p2
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/oelm-pretrain-v2/phase2_%j.out
#SBATCH --error=logs/oelm-pretrain-v2/phase2_%j.err

set -euo pipefail

METHOD=${1:-baseline}
TASK=${2:-sst2}
SEED=${3:-42}
CHECKPOINT=${4:-}
PROJECT_ROOT=${OELM_PROJECT_ROOT:-/projects/LlamaFactory/OELM-Pretrain}
CONDA_ENV=${OELM_CONDA_ENV:-/projects/LlamaFactory/miniconda3/envs/oelm}

if [ -z "${CHECKPOINT}" ]; then
  echo "Usage: $0 [baseline|qk_only|qk_ffn] [sst2|ag_news] [seed] <checkpoint_dir>"
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs/oelm-pretrain-v2"
mkdir -p "${PROJECT_ROOT}/outputs/oelm-pretrain-v2"
mkdir -p "${PROJECT_ROOT}/results/oelm-pretrain-v2"

cd "${PROJECT_ROOT}"
module load Miniforge3
source activate "${CONDA_ENV}"

python experiments/oelm-pretrain-v2/scripts/evaluate_downstream_v2.py \
  --phase phase2_quick \
  --config_path experiments/oelm-pretrain-v2/configs/quick_eval_tasks.json \
  --checkpoint "${CHECKPOINT}" \
  --task "${TASK}" \
  --setting setting_a \
  --method "${METHOD}" \
  --seed "${SEED}" \
  --output_root "${PROJECT_ROOT}/results/oelm-pretrain-v2" \
  --max_seq_len 512 \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --weight_decay 0.01 \
  --num_epochs 10 \
  --max_steps 500 \
  --warmup_steps 50 \
  --patience 3
