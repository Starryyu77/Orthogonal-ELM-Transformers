#!/bin/bash -l
#SBATCH --job-name=oelm-v2-p5
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/oelm-pretrain-v2/phase5_%j.out
#SBATCH --error=logs/oelm-pretrain-v2/phase5_%j.err

set -euo pipefail

METHOD=${1:-baseline}
TASK=${2:-sst2}
SETTING=${3:-setting_a}
SEED=${4:-42}
CHECKPOINT=${5:-}
PROJECT_ROOT=${OELM_PROJECT_ROOT:-/projects/LlamaFactory/OELM-Pretrain}
CONDA_ENV=${OELM_CONDA_ENV:-/projects/LlamaFactory/miniconda3/envs/oelm}

if [ -z "${CHECKPOINT}" ]; then
  echo "Usage: $0 [baseline|qk_only|qk_ffn] [task] [setting] [seed] <checkpoint_dir>"
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs/oelm-pretrain-v2"
mkdir -p "${PROJECT_ROOT}/outputs/oelm-pretrain-v2"
mkdir -p "${PROJECT_ROOT}/results/oelm-pretrain-v2"

cd "${PROJECT_ROOT}"
module load Miniforge3
source activate "${CONDA_ENV}"

python experiments/oelm-pretrain-v2/scripts/evaluate_downstream_v2.py \
  --phase phase5_broad_eval \
  --config_path experiments/oelm-pretrain-v2/configs/broad_eval_tasks.json \
  --checkpoint "${CHECKPOINT}" \
  --task "${TASK}" \
  --setting "${SETTING}" \
  --method "${METHOD}" \
  --seed "${SEED}" \
  --output_root "${PROJECT_ROOT}/results/oelm-pretrain-v2" \
  --max_seq_len 512 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --num_epochs 3 \
  --max_steps 10000 \
  --warmup_steps 200 \
  --patience 3
