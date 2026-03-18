#!/bin/bash
#SBATCH --job-name=xnli_baseline
#SBATCH --partition=Pro
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=logs/xnli_baseline_%j.out
#SBATCH --error=logs/xnli_baseline_%j.err

# XNLI - Baseline Fine-tuning Script
# Phase 6 Multi-Dataset Validation

cd /projects/LlamaFactory/OELM-Pretrain

# Load environment
module load Miniforge3
source activate

# Create output directory
mkdir -p outputs/phase6_multidata/xnli/baseline
mkdir -p logs

echo "=========================================="
echo "XNLI - Baseline Fine-tuning"
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Run fine-tuning
python scripts/finetune_multidata.py \
    --pretrained_path outputs/pretrain/baseline/final_model.pt \
    --dataset xnli \
    --output_dir outputs/phase6_multidata/xnli/baseline \
    --batch_size 16 \
    --num_epochs 3 \
    --lr 3e-4 \
    --max_seq_len 512 \
    --gpu 0

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
