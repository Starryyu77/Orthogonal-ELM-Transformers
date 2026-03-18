#!/bin/bash
#SBATCH --job-name=oelm-finetune-qk-ffn
#SBATCH --time=4:00:00
#SBATCH --gpus=pro6000:1
#SBATCH --output=logs/finetune_qk_ffn-%j.out
#SBATCH --error=logs/finetune_qk_ffn-%j.err

# Load environment
module load Miniforge3
source activate

cd /projects/LlamaFactory/OELM-Pretrain

# Find best checkpoint
PRETRAINED_PATH=$(ls -t outputs/pretrain/oelm_qk_ffn/best/pytorch_model.pt 2>/dev/null | head -1)

if [ -z "$PRETRAINED_PATH" ]; then
    echo "Error: No pretrained model found!"
    exit 1
fi

echo "========================================"
echo "Fine-tuning OELM-QK-FFN Model"
echo "Pretrained: $PRETRAINED_PATH"
echo "========================================"

# Run fine-tuning with frozen Q/K
python scripts/finetune_from_pretrain.py \
    --pretrained_path $PRETRAINED_PATH \
    --dataset imdb \
    --freeze_qk \
    --batch_size 16 \
    --num_epochs 3 \
    --lr 1e-3 \
    --output_dir ./outputs/finetune/oelm_qk_ffn_imdb \
    --gpu 0

echo "========================================"
echo "End time: $(date)"
echo "========================================"