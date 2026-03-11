#!/bin/bash
#SBATCH --job-name=sst2_oelm_qk_ffn
#SBATCH --partition=Pro
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sst2_oelm_qk_ffn_%j.out
#SBATCH --error=logs/sst2_oelm_qk_ffn_%j.err

# SST-2 - OELM-QK-FFN Fine-tuning Script
# Phase 6 Multi-Dataset Validation

cd /projects/LlamaFactory/OELM-Pretrain

# Load environment
module load Miniforge3
source activate

# Create output directory
mkdir -p outputs/phase6_multidata/sst2/oelm_qk_ffn
mkdir -p logs

echo "=========================================="
echo "SST-2 - OELM-QK-FFN Fine-tuning"
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Run fine-tuning
python scripts/finetune_multidata.py \
    --pretrained_path outputs/pretrain/oelm_qk_ffn/final_model.pt \
    --dataset sst2 \
    --output_dir outputs/phase6_multidata/sst2/oelm_qk_ffn \
    --batch_size 32 \
    --num_epochs 3 \
    --lr 1e-3 \
    --freeze_qk \
    --max_seq_len 128 \
    --gpu 0

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
