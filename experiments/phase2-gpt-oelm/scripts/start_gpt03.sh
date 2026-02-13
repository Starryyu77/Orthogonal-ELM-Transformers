#!/bin/bash
# GPT-03: TinyStories OELM-Random Ablation Experiment

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# 使用 GPU 2 (GPT-01 完成后)
export CUDA_VISIBLE_DEVICES=2

# 会话名称
SESSION_NAME="gpt03_random"

# 检查会话是否已存在
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already exists!"
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 1
fi

# 创建新的 tmux 会话
tmux new-session -d -s "$SESSION_NAME" -n "training"

# 发送训练命令
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR && echo 'Starting GPT-03: TinyStories OELM-Random'" C-m
tmux send-keys -t "$SESSION_NAME" "python3 scripts/train_v2.py \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --model_type oelm_random \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --dataset tinystories \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --max_lr 1e-3 \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --max_steps 100000 \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --out_dir outputs/GPT-03_oelm_random \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --val_interval 1000 \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --use_amp" C-m

echo "✓ GPT-03 (OELM-Random) started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME    # 实时查看"
echo "  tmux capture-pane -t $SESSION_NAME -p | tail -5  # 查看进度"
echo ""
