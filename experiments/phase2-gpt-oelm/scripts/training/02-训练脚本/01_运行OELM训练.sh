#!/bin/bash
# Run Orthogonal ELM Transformer training

# Default configuration
DATASET=${DATASET:-"tinystories"}
MODEL_SIZE=${MODEL_SIZE:-"small"}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_STEPS=${MAX_STEPS:-100000}
ORTHO_METHOD=${ORTHO_METHOD:-"qr"}

# Data path
DATA_PATH="data/${DATASET}/train.bin"

# Output directory
OUT_DIR="out/oelm_${MODEL_SIZE}_${DATASET}"

echo "=========================================="
echo "Training Orthogonal ELM Transformer"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model Size: $MODEL_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Max Steps: $MAX_STEPS"
echo "Ortho Method: $ORTHO_METHOD"
echo "Output: $OUT_DIR"
echo "=========================================="

# Run training
python train.py \
    --model_type oelm \
    --ortho_method $ORTHO_METHOD \
    --data_path $DATA_PATH \
    --out_dir $OUT_DIR \
    --batch_size $BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --use_wandb \
    --wandb_project orthogonal-elm \
    --wandb_run_name "oelm_${MODEL_SIZE}_${DATASET}" \
    --log_interval 100 \
    --val_interval 1000 \
    --save_interval 5000

echo "Training complete!"
