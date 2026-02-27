#!/bin/bash
# AG News Baseline 实验启动脚本
# 用法: ./run_agnews_baseline.sh [GPU_ID]

GPU_ID=${1:-0}

echo "=========================================="
echo "AG News Baseline 实验"
echo "GPU: $GPU_ID"
echo "开始时间: $(date)"
echo "=========================================="

# 项目目录
PROJECT_DIR="$HOME/Orthogonal_ELM_Transformers/Train"
cd $PROJECT_DIR

# 激活环境
source ~/projects/oelm/venv/bin/activate

# 输出目录
OUTPUT_DIR="$PROJECT_DIR/outputs/AGNews_baseline"
mkdir -p $OUTPUT_DIR

# 启动训练
echo "启动训练..."
python experiments/phase4-gpt-classification/scripts/train_classification.py \
    --model_type baseline \
    --dataset ag_news \
    --num_classes 4 \
    --d_model 512 \
    --num_layers 6 \
    --num_heads 8 \
    --d_ff 2048 \
    --max_seq_len 512 \
    --batch_size 16 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --output_dir $OUTPUT_DIR \
    --device cuda:$GPU_ID \
    --log_interval 100

echo "=========================================="
echo "实验完成"
echo "结束时间: $(date)"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
