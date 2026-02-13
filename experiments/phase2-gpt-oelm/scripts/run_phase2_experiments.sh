#!/bin/bash
# Phase 2: GPT OELM v2 Experiments
# Head-wise Orthogonal Initialization vs Baseline

set -e

PROJECT_DIR="/usr1/home/s125mdg43_10/Orthogonal_ELM_Transformers/Train/gpt-oelm-project"
OUTPUT_BASE="${PROJECT_DIR}/outputs"

echo "========================================"
echo "Phase 2: GPT OELM v2 Experiments"
echo "========================================"
echo ""

# Create output directories
mkdir -p ${OUTPUT_BASE}/{tinystories,openwebtext,wikitext103}/{baseline,oelm_v2}

# ========================================
# Dataset 1: TinyStories
# ========================================
echo "========================================"
echo "TinyStories Experiments"
echo "========================================"
echo ""

# Check if data exists
if [ ! -f "${PROJECT_DIR}/data/tinystories/train.bin" ]; then
    echo "⚠️  TinyStories data not found. Preparing..."
    cd ${PROJECT_DIR}/data
    python3 prepare_data.py --dataset tinystories
fi

# Baseline (GPU 2)
echo "Launching TinyStories Baseline on GPU 2..."
tmux new-session -d -s gpt_tinystories_baseline
tmux send-keys -t gpt_tinystories_baseline "cd ${PROJECT_DIR}" C-m
tmux send-keys -t gpt_tinystories_baseline "export CUDA_VISIBLE_DEVICES=2" C-m
tmux send-keys -t gpt_tinystories_baseline "python3 scripts/train_v2.py \\" C-m
tmux send-keys -t gpt_tinystories_baseline "  --model_type baseline \\" C-m
tmux send-keys -t gpt_tinystories_baseline "  --dataset tinystories \\" C-m
tmux send-keys -t gpt_tinystories_baseline "  --max_steps 10000 \\" C-m
tmux send-keys -t gpt_tinystories_baseline "  --batch_size 16 \\" C-m
tmux send-keys -t gpt_tinystories_baseline "  --max_lr 3e-4 \\" C-m
tmux send-keys -t gpt_tinystories_baseline "  --out_dir ${OUTPUT_BASE}/tinystories/baseline \\" C-m
tmux send-keys -t gpt_tinystories_baseline "  2>&1 | tee ${OUTPUT_BASE}/tinystories/baseline/train.log" C-m

# OELM v2 (GPU 3)
echo "Launching TinyStories OELM v2 on GPU 3..."
tmux new-session -d -s gpt_tinystories_oelm
tmux send-keys -t gpt_tinystories_oelm "cd ${PROJECT_DIR}" C-m
tmux send-keys -t gpt_tinystories_oelm "export CUDA_VISIBLE_DEVICES=3" C-m
tmux send-keys -t gpt_tinystories_oelm "python3 scripts/train_v2.py \\" C-m
tmux send-keys -t gpt_tinystories_oelm "  --model_type oelm_v2 \\" C-m
tmux send-keys -t gpt_tinystories_oelm "  --dataset tinystories \\" C-m
tmux send-keys -t gpt_tinystories_oelm "  --max_steps 10000 \\" C-m
tmux send-keys -t gpt_tinystories_oelm "  --batch_size 16 \\" C-m
tmux send-keys -t gpt_tinystories_oelm "  --max_lr 3e-4 \\" C-m
tmux send-keys -t gpt_tinystories_oelm "  --out_dir ${OUTPUT_BASE}/tinystories/oelm_v2 \\" C-m
tmux send-keys -t gpt_tinystories_oelm "  2>&1 | tee ${OUTPUT_BASE}/tinystories/oelm_v2/train.log" C-m

echo "✓ TinyStories experiments launched"
echo ""

# ========================================
# Summary
# ========================================
echo "========================================"
echo "Experiments Launched!"
echo "========================================"
echo ""
echo "Active sessions:"
echo "  - gpt_tinystories_baseline  (GPU 2)"
echo "  - gpt_tinystories_oelm      (GPU 3)"
echo ""
echo "Next steps after completion:"
echo "  1. Run OpenWebText experiments"
echo "  2. Run WikiText-103 experiments"
echo "  3. Run ablation (OELM-Random)"
echo ""
