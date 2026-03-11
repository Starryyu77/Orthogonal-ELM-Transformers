#!/bin/bash
#SBATCH --job-name=oelm-finetune-baseline
#SBATCH --time=4:00:00
#SBATCH --gpus=pro6000:1
#SBATCH --output=logs/finetune_baseline-%j.out
#SBATCH --error=logs/finetune_baseline-%j.err

# Load environment
module load Miniforge3
source activate

# Set working directory
cd /projects/LlamaFactory/OELM-Pretrain

# Find best checkpoint
PRETRAINED_PATH=$(ls -t outputs/pretrain/baseline/best/pytorch_model.pt 2>/dev/null | head -1)

if [ -z "$PRETRAINED_PATH" ]; then
    echo "Error: No pretrained model found!"
    echo "Looking for: outputs/pretrain/baseline/best/pytorch_model.pt"
    exit 1
fi

echo "========================================"
echo "Fine-tuning Baseline Model"
echo "Pretrained: $PRETRAINED_PATH"
echo "========================================"

# Run fine-tuning for IMDB
python scripts/finetune_from_pretrain.py \
    --pretrained_path $PRETRAINED_PATH \
    --dataset imdb \
    --batch_size 16 \
    --num_epochs 3 \
    --lr 3e-4 \
    --output_dir ./outputs/finetune/baseline_imdb \
    --gpu 0

echo "========================================"
echo "End time: $(date)"
echo "========================================"