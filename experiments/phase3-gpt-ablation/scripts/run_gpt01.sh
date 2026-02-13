#!/bin/bash
# =============================================================================
# GPT-01: TinyStories Baseline
# =============================================================================
# 实验描述:
#   - 数据集: TinyStories (短篇故事数据集)
#   - 模型: GPT Medium-512 (6层, 8头, 512维)
#   - 方法: 标准训练 (所有参数可训练)
#   - 学习率: 3e-4 (标准transformer学习率)
#
# 预期结果:
#   - Final PPL: ~4.27
#   - 训练时间: ~4.5小时
#   - 此实验作为后续OELM实验的基准
#
# 启动命令: ./run_gpt01.sh [gpu_id]
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

# GPU选择 (默认GPU 2)
GPU_ID="${1:-2}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 实验配置
EXP_ID="GPT-01"
DATASET="tinystories"
MODEL_TYPE="baseline"
LR="3e-4"
MAX_STEPS=100000
GPU_NAME="gpu${GPU_ID}"

SESSION_NAME="${EXP_ID,,}_${DATASET}_${MODEL_TYPE}"
OUT_DIR="outputs/${EXP_ID}_${DATASET}_${MODEL_TYPE}"

echo "=========================================="
echo "启动实验: $EXP_ID"
echo "=========================================="
echo "配置:"
echo "  数据集: $DATASET"
echo "  模型: $MODEL_TYPE"
echo "  学习率: $LR"
echo "  步数: $MAX_STEPS"
echo "  GPU: $GPU_NAME"
echo "  输出: $OUT_DIR"
echo "=========================================="

# 检查tmux会话
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "警告: 会话 $SESSION_NAME 已存在"
    echo "使用: tmux attach -t $SESSION_NAME"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUT_DIR"

# 启动tmux会话
tmux new-session -d -s "$SESSION_NAME" -n "training"

# 发送训练命令
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR && echo '[$(date)] 开始训练 $EXP_ID'" C-m
tmux send-keys -t "$SESSION_NAME" "python3 scripts/train_v2.py \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --model_type $MODEL_TYPE \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --dataset $DATASET \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --max_lr $LR \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --max_steps $MAX_STEPS \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --out_dir $OUT_DIR \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --val_interval 1000 \\" C-m
tmux send-keys -t "$SESSION_NAME" "  --use_amp \\" C-m
tmux send-keys -t "$SESSION_NAME" "  2>&1 | tee $OUT_DIR/training.log" C-m

echo ""
echo "✓ $EXP_ID 已启动在 $GPU_NAME"
echo ""
echo "监控命令:"
echo "  实时查看: tmux attach -t $SESSION_NAME"
echo "  查看日志: tail -f $OUT_DIR/training.log"
echo "  查看进度: tmux capture-pane -t $SESSION_NAME -p | tail -5"
echo ""
echo "预期完成时间: ~4.5小时"
