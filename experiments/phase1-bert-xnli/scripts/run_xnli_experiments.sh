#!/bin/bash
# XNLI Experiments - Phase 1
# BERT Baseline vs OELM-Freeze on XNLI dataset

set -e

echo "========================================"
echo "XNLI Experiments - Phase 1"
echo "========================================"
echo ""

# Configuration
PROJECT_DIR="/usr1/home/s125mdg43_10/Orthogonal_ELM_Transformers/Train/bert-reservoir-project"
OUTPUT_BASE="/usr1/home/s125mdg43_10/Orthogonal_ELM_Transformers/Train/bert-reservoir-project/outputs/xnli"

# Create output directories
mkdir -p ${OUTPUT_BASE}/baseline
mkdir -p ${OUTPUT_BASE}/oelm_freeze

echo "Output directories created:"
echo "  - ${OUTPUT_BASE}/baseline"
echo "  - ${OUTPUT_BASE}/oelm_freeze"
echo ""

# ========================================
# Experiment 1: XNLI Baseline (GPU 2)
# ========================================
echo "========================================"
echo "Launching XNLI Baseline on GPU 2"
echo "========================================"
echo ""

tmux new-session -d -s xnli_baseline

# Send commands to the session
tmux send-keys -t xnli_baseline "cd ${PROJECT_DIR}" C-m
tmux send-keys -t xnli_baseline "export CUDA_VISIBLE_DEVICES=2" C-m
tmux send-keys -t xnli_baseline "export PYTHONPATH=${PROJECT_DIR}:$PYTHONPATH" C-m

tmux send-keys -t xnli_baseline "python3 models/train_bert.py \\" C-m
tmux send-keys -t xnli_baseline "  --dataset xnli \\" C-m
tmux send-keys -t xnli_baseline "  --freeze_mode false \\" C-m
tmux send-keys -t xnli_baseline "  --init_method normal \\" C-m
tmux send-keys -t xnli_baseline "  --lr 2e-5 \\" C-m
tmux send-keys -t xnli_baseline "  --batch_size 16 \\" C-m
tmux send-keys -t xnli_baseline "  --epochs 3 \\" C-m
tmux send-keys -t xnli_baseline "  --max_length 128 \\" C-m
tmux send-keys -t xnli_baseline "  --validate_steps 500 \\" C-m
tmux send-keys -t xnli_baseline "  --output_dir ${OUTPUT_BASE}/baseline \\" C-m
tmux send-keys -t xnli_baseline "  --seed 42 \\" C-m
tmux send-keys -t xnli_baseline "  2>&1 | tee ${OUTPUT_BASE}/baseline/training.log" C-m

echo "✓ XNLI Baseline launched in tmux session 'xnli_baseline'"
echo ""

# ========================================
# Experiment 2: XNLI OELM-Freeze (GPU 3)
# ========================================
echo "========================================"
echo "Launching XNLI OELM-Freeze on GPU 3"
echo "========================================"
echo ""

tmux new-session -d -s xnli_oelm

# Send commands to the session
tmux send-keys -t xnli_oelm "cd ${PROJECT_DIR}" C-m
tmux send-keys -t xnli_oelm "export CUDA_VISIBLE_DEVICES=3" C-m
tmux send-keys -t xnli_oelm "export PYTHONPATH=${PROJECT_DIR}:$PYTHONPATH" C-m

tmux send-keys -t xnli_oelm "python3 models/train_bert.py \\" C-m
tmux send-keys -t xnli_oelm "  --dataset xnli \\" C-m
tmux send-keys -t xnli_oelm "  --freeze_mode true \\" C-m
tmux send-keys -t xnli_oelm "  --init_method orthogonal \\" C-m
tmux send-keys -t xnli_oelm "  --lr 1e-4 \\" C-m
tmux send-keys -t xnli_oelm "  --batch_size 16 \\" C-m
tmux send-keys -t xnli_oelm "  --epochs 3 \\" C-m
tmux send-keys -t xnli_oelm "  --max_length 128 \\" C-m
tmux send-keys -t xnli_oelm "  --validate_steps 500 \\" C-m
tmux send-keys -t xnli_oelm "  --output_dir ${OUTPUT_BASE}/oelm_freeze \\" C-m
tmux send-keys -t xnli_oelm "  --seed 42 \\" C-m
tmux send-keys -t xnli_oelm "  2>&1 | tee ${OUTPUT_BASE}/oelm_freeze/training.log" C-m

echo "✓ XNLI OELM-Freeze launched in tmux session 'xnli_oelm'"
echo ""

# ========================================
# Summary
# ========================================
echo "========================================"
echo "Experiments Launched Successfully!"
echo "========================================"
echo ""
echo "Active tmux sessions:"
echo "  - xnli_baseline  (GPU 2, Baseline)"
echo "  - xnli_oelm      (GPU 3, OELM-Freeze)"
echo ""
echo "Monitor commands:"
echo "  tmux attach -t xnli_baseline   # View Baseline training"
echo "  tmux attach -t xnli_oelm       # View OELM-Freeze training"
echo ""
echo "Check GPU status:"
echo "  nvidia-smi"
echo ""
echo "Logs location:"
echo "  ${OUTPUT_BASE}/baseline/training.log"
echo "  ${OUTPUT_BASE}/oelm_freeze/training.log"
echo ""
echo "Timing data will be saved to:"
echo "  ${OUTPUT_BASE}/baseline/timing_stats.json"
echo "  ${OUTPUT_BASE}/oelm_freeze/timing_stats.json"
echo ""
