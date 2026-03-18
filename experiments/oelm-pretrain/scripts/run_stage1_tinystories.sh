#!/bin/bash
#SBATCH --job-name=oelm-stage1-tinystories
#SBATCH --partition=cluster02
#SBATCH --gpus=pro-6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

# OELM Stage 1: TinyStories Quick Validation
# Usage: sbatch scripts/run_stage1_tinystories.sh [baseline|oelm_qk|oelm_qk_ffn]

METHOD=${1:-baseline}  # Default to baseline

echo "=========================================="
echo "Stage 1: TinyStories Pretraining"
echo "Method: $METHOD"
echo "Started: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

cd /projects/LlamaFactory/OELM-Pretrain

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate oelm

# Set method-specific arguments
case $METHOD in
    baseline)
        FREEZE_QK="false"
        FREEZE_FFN="false"
        LR="5e-4"
        ;;
    oelm_qk)
        FREEZE_QK="true"
        FREEZE_FFN="false"
        LR="5e-4"
        ;;
    oelm_qk_ffn)
        FREEZE_QK="true"
        FREEZE_FFN="true"
        LR="5e-4"
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Usage: $0 [baseline|oelm_qk|oelm_qk_ffn]"
        exit 1
        ;;
esac

# Run training
python scripts/pretrain.py \
    --dataset tinystories \
    --model_name gpt2 \
    --max_steps 10000 \
    --batch_size 32 \
    --seq_length 512 \
    --learning_rate $LR \
    --warmup_steps 1000 \
    --freeze_qk $FREEZE_QK \
    --freeze_ffn $FREEZE_FFN \
    --output_dir outputs/stage1_tinystories/$METHOD \
    --save_steps 2000 \
    --eval_steps 500 \
    --logging_steps 100 \
    --fp16 \
    --gradient_accumulation_steps 1

echo "=========================================="
echo "Completed: $(date)"
echo "Output: outputs/stage1_tinystories/$METHOD"
echo "=========================================="
