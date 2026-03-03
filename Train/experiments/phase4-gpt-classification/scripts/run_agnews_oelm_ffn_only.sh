#!/bin/bash
# AG News OELM-FFN-only 实验启动脚本
# 用法: ./run_agnews_oelm_ffn_only.sh [GPU_ID] [LR_INDEX]
#   GPU_ID: GPU编号 (默认0)
#   LR_INDEX: 学习率索引 0=5e-4, 1=1e-3, 2=3e-3 (默认0)

GPU_ID=${1:-0}
LR_INDEX=${2:-0}

# 学习率选项
LEARNING_RATES=("5e-4" "1e-3" "3e-3")
LR_NAMES=("5e-4" "1e-3" "3e-3")

if [ "$LR_INDEX" -lt 0 ] || [ "$LR_INDEX" -gt 2 ]; then
    echo "错误: LR_INDEX必须在0-2之间"
    echo "  0: lr=5e-4"
    echo "  1: lr=1e-3"
    echo "  2: lr=3e-3"
    exit 1
fi

LR=${LEARNING_RATES[$LR_INDEX]}
LR_NAME=${LR_NAMES[$LR_INDEX]}

echo "=========================================="
echo "AG News OELM-FFN-only 实验"
echo "GPU: $GPU_ID"
echo "学习率: $LR"
echo "开始时间: $(date)"
echo "=========================================="

# 项目目录
PROJECT_DIR="/projects/Orthogonal_ELM_Transformers/Train"
cd $PROJECT_DIR

# 激活环境
source ~/projects/oelm/venv/bin/activate

# 输出目录
OUTPUT_DIR="$PROJECT_DIR/outputs_phase4/AGNEWS_oelm_ffn_only_lr${LR_NAME}"
mkdir -p $OUTPUT_DIR

# 启动训练
echo "启动训练..."
python experiments/phase4-gpt-classification/scripts/train_classification.py \
    --model_type oelm_ffn_only \
    --dataset ag_news \
    --num_classes 4 \
    --d_model 512 \
    --num_layers 6 \
    --num_heads 8 \
    --d_ff 2048 \
    --max_seq_len 512 \
    --batch_size 16 \
    --num_epochs 3 \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR \
    --device cuda:$GPU_ID \
    --log_interval 100 \
    --seed 42

echo "=========================================="
echo "实验完成"
echo "结束时间: $(date)"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
