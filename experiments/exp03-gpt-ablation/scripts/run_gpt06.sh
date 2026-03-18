#!/bin/bash
# =============================================================================
# GPT-06: WikiText-103 Baseline
# =============================================================================
# 实验描述:
#   - 数据集: WikiText-103 (大型维基百科文本数据集)
#   - 模型: GPT Medium-512 (6层, 8头, 512维)
#   - 方法: 标准训练 (所有参数可训练)
#   - 学习率: 3e-4
#
# 数据集特点:
#   - 比OpenWebText更大
#   - 知识密集型文本
#   - 用于测试方法在知识密集型任务上的效果
#
# 预期配置:
#   - Max Steps: 200,000
#   - 预计训练时间: ~12小时
#
# 状态: 待启动
#
# 启动命令: ./run_gpt06.sh [gpu_id]
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

# GPU选择 (默认GPU 2)
GPU_ID="${1:-2}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 实验配置
EXP_ID="GPT-06"
DATASET="wikitext103"
MODEL_TYPE="baseline"
LR="3e-4"
MAX_STEPS=200000
GPU_NAME="gpu${GPU_ID}"

SESSION_NAME="${EXP_ID,,}_${DATASET}_${MODEL_TYPE}"
OUT_DIR="outputs/${EXP_ID}_${DATASET}_${MODEL_TYPE}"

echo "=========================================="
echo "启动实验: $EXP_ID"
echo "=========================================="
echo "配置:"
echo "  数据集: $DATASET (WikiText-103)"
echo "  模型: $MODEL_TYPE (标准训练)"
echo "  学习率: $LR"
echo "  步数: $MAX_STEPS"
echo "  GPU: $GPU_NAME"
echo "  输出: $OUT_DIR"
echo ""
echo "注意: 这是Phase 3最后一个Baseline实验"
echo "=========================================="

# 数据准备检查
echo ""
echo "检查WikiText-103数据..."
if [ ! -f "$PROJECT_DIR/experiments/phase2-gpt-oelm/data/wikitext103/train.bin" ]; then
    echo "⚠️  警告: WikiText-103数据未找到"
    echo "请先运行数据准备:"
    echo "  python3 data/prepare_data.py --dataset wikitext103"
    echo ""
    read -p "是否继续? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

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

# 发送训练命令 (包含CUDA_VISIBLE_DEVICES)
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR && export CUDA_VISIBLE_DEVICES=$GPU_ID && echo '[$(date)] 开始训练 $EXP_ID on GPU $GPU_ID'" C-m
tmux send-keys -t "$SESSION_NAME" "python3 experiments/phase2-gpt-oelm/scripts/train_v2.py \\" C-m
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
echo "预计完成时间: ~12小时"
echo "对比实验: GPT-07 (WikiText-103 OELM-Freeze)"
