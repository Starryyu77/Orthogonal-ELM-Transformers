#!/bin/bash
# GPT-05: OpenWebText OELM-Freeze

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# 使用 GPU 3
export CUDA_VISIBLE_DEVICES=3

SESSION_NAME="gpt05_openwebtext_oelm"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already exists!"
    exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -n "training"

tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR && echo 'Starting GPT-05: OpenWebText OELM-Freeze'" C-m
tmux send-keys -t "$SESSION_NAME" "python3 scripts/train_v2.py \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --model_type oelm_v2 \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --dataset openwebtext \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --max_lr 1e-3 \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --max_steps 150000 \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --out_dir outputs/GPT-05_openwebtext_oelm \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --val_interval 1000 \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --use_amp" C-m

echo "✓ GPT-05 (OpenWebText OELM-Freeze) started"
echo "Attach: tmux attach -t $SESSION_NAME"
