#!/bin/bash
# XNLI OELM-Freeze Restart with Gradient Accumulation

set -e

PROJECT_DIR="/usr1/home/s125mdg43_10/Orthogonal_ELM_Transformers/Train/bert-reservoir-project"
OUTPUT_BASE="${PROJECT_DIR}/outputs/xnli"

mkdir -p ${OUTPUT_BASE}/oelm_freeze

echo "========================================"
echo "Restarting XNLI OELM-Freeze"
echo "Configuration:"
echo "  - Batch size: 8"
echo "  - Gradient accumulation: 2"
echo "  - Effective batch size: 16"
echo "  - GPU: 3"
echo "========================================"
echo ""

tmux new-session -d -s xnli_oelm

tmux send-keys -t xnli_oelm "cd ${PROJECT_DIR}" C-m
tmux send-keys -t xnli_oelm "export CUDA_VISIBLE_DEVICES=3" C-m
tmux send-keys -t xnli_oelm "export PYTHONPATH=${PROJECT_DIR}:$PYTHONPATH" C-m

tmux send-keys -t xnli_oelm "python3 models/train_bert.py \\" C-m
tmux send-keys -t xnli_oelm "  --dataset xnli \\" C-m
tmux send-keys -t xnli_oelm "  --freeze_mode true \\" C-m
tmux send-keys -t xnli_oelm "  --init_method orthogonal \\" C-m
tmux send-keys -t xnli_oelm "  --lr 1e-4 \\" C-m
tmux send-keys -t xnli_oelm "  --batch_size 8 \\" C-m
tmux send-keys -t xnli_oelm "  --gradient_accumulation_steps 2 \\" C-m
tmux send-keys -t xnli_oelm "  --epochs 3 \\" C-m
tmux send-keys -t xnli_oelm "  --max_length 128 \\" C-m
tmux send-keys -t xnli_oelm "  --validate_steps 500 \\" C-m
tmux send-keys -t xnli_oelm "  --output_dir ${OUTPUT_BASE}/oelm_freeze \\" C-m
tmux send-keys -t xnli_oelm "  --seed 42 \\" C-m
tmux send-keys -t xnli_oelm "  2>&1 | tee ${OUTPUT_BASE}/oelm_freeze/training_restart.log" C-m

echo "✓ XNLI OELM-Freeze restarted in tmux session 'xnli_oelm'"
echo ""
echo "Effective batch size: 8 * 2 = 16"
echo "Monitor: tmux attach -t xnli_oelm"
