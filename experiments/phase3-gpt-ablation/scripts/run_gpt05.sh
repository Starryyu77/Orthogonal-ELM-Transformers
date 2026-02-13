#!/bin/bash
# =============================================================================
# GPT-05: OpenWebText OELM-Freeze
# =============================================================================
# 实验描述:
#   - 数据集: OpenWebText
#   - 模型: GPT Medium-512
#   - 方法: OELM (冻结Q/K投影, 正交初始化)
#   - 学习率: 1e-3
#
# 关键技术:
#   - Q/K投影层使用分头正交初始化 (head-wise orthogonal)
#   - Q/K权重在训练中冻结 (freeze)
#   - V/O保持可训练
#
# 历史结果 (参考):
#   - Final PPL: 54.29
#   - 与Baseline差距: +14.9%
#   - 训练时间: ~9小时
#
# 启动命令: ./run_gpt05.sh [gpu_id]
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

# GPU选择 (默认GPU 3)
GPU_ID="${1:-3}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 实验配置
EXP_ID="GPT-05"
DATASET="openwebtext"
MODEL_TYPE="oelm_v2"
LR="1e-3"
MAX_STEPS=150000
GPU_NAME="gpu${GPU_ID}"

SESSION_NAME="${EXP_ID,,}_${DATASET}_${MODEL_TYPE}"
OUT_DIR="outputs/${EXP_ID}_${DATASET}_${MODEL_TYPE}"

echo "=========================================="
echo "启动实验: $EXP_ID"
echo "=========================================="
echo "配置:"
echo "  数据集: $DATASET (OpenWebText)"
echo "  模型: $MODEL_TYPE (OELM-Freeze)"
echo "  学习率: $LR (OELM使用更高学习率)"
echo "  步数: $MAX_STEPS"
echo "  GPU: $GPU_NAME"
echo "  输出: $OUT_DIR"
echo ""
echo "OELM特性:"
echo "  - Q/K正交初始化: 是"
echo "  - Q/K冻结: 是"
echo "=========================================="

# 检查tmux会话
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "警告: 会话 $SESSION_NAME 已存在"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUT_DIR"

# 启动tmux会话
tmux new-session -d -s "$SESSION_NAME" -n "training"

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
echo "对比基准: GPT-04 (OpenWebText Baseline PPL: 47.24)"
echo "目标: 验证在复杂数据集上OELM的表现"
echo "预计完成时间: ~9小时"
