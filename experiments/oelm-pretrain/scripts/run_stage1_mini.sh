#!/bin/bash -l
#SBATCH --job-name=oelm-s1-mini
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=logs/stage1_mini_%j.out
#SBATCH --error=logs/stage1_mini_%j.err

# OELM Stage 1: TinyStories Quick Validation with Mini Model (42M params)
# Usage: sbatch run_stage1_mini.sh [baseline|oelm_qk|oelm_qk_ffn]

METHOD=${1:-baseline}

echo "=========================================="
echo "Stage 1: TinyStories with Mini Model"
echo "Model: GPT-2 Mini (42M params, d=512, l=6, h=8)"
echo "Method: $METHOD"
echo "Started: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

cd /projects/LlamaFactory/OELM-Pretrain

# Load environment
module load Miniforge3
source activate
conda activate oelm

# Create output directory
mkdir -p outputs/stage1_tinystories_mini
mkdir -p logs

# Set method-specific arguments
case $METHOD in
    baseline)
        FREEZE_QK=""
        FREEZE_FFN=""
        echo "Running Baseline (all trainable)"
        ;;
    oelm_qk)
        FREEZE_QK="--freeze_qk"
        FREEZE_FFN=""
        echo "Running OELM-QK (freeze Q/K)"
        ;;
    oelm_qk_ffn)
        FREEZE_QK="--freeze_qk"
        FREEZE_FFN="--freeze_ffn"
        echo "Running OELM-QK-FFN (freeze Q/K + FFN)"
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Usage: $0 [baseline|oelm_qk|oelm_qk_ffn]"
        exit 1
        ;;
esac

OUTPUT_DIR="outputs/stage1_tinystories_mini/$METHOD"

echo "Output directory: $OUTPUT_DIR"
echo "Starting training..."
echo ""

# Run training with mini model
python experiments/oelm-pretrain/scripts/pretrain_mini.py \
    --dataset tinystories \
    --max_steps 10000 \
    --batch_size 32 \
    --seq_length 512 \
    --learning_rate 5e-4 \
    --warmup_steps 1000 \
    $FREEZE_QK \
    $FREEZE_FFN \
    --output_dir $OUTPUT_DIR \
    --save_steps 2000 \
    --logging_steps 100

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Print final results if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Final checkpoint: $OUTPUT_DIR/final"
    echo "To check results:"
    echo "  ls -lh $OUTPUT_DIR/"
    echo "  cat $OUTPUT_DIR/final/model_config.json"
    echo "  tail -20 logs/stage1_mini_*.out"
fi

exit $EXIT_CODE
