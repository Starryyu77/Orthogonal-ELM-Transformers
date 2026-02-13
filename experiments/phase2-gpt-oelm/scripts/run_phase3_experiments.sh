#!/bin/bash
# =============================================================================
# Phase 3: GPT OELM Ablation Experiments
# =============================================================================
# 实验设计:
# - TinyStories: Baseline + OELM-Freeze + OELM-Random
# - OpenWebText: Baseline + OELM-Freeze
# - WikiText-103: Baseline + OELM-Freeze
#
# 每个实验都包含完整的 CUDA synchronized 计时
# =============================================================================

set -e

# 配置
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"
OUTPUT_BASE="$PROJECT_DIR/outputs"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train_v2.py"

# 默认配置
DEFAULT_D_MODEL=512
DEFAULT_NUM_LAYERS=6
DEFAULT_NUM_HEADS=8
DEFAULT_D_FF=2048
DEFAULT_SEQ_LEN=512
DEFAULT_BATCH_SIZE=8
DEFAULT_MAX_STEPS=100000

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

show_help() {
    echo "Phase 3: GPT OELM Ablation Experiments"
    echo ""
    echo "Usage: $0 <experiment_id>"
    echo ""
    echo "Experiments:"
    echo "  GPT-01    TinyStories Baseline (lr=3e-4)"
    echo "  GPT-02    TinyStories OELM-Freeze (lr=1e-3, orthogonal init)"
    echo "  GPT-03    TinyStories OELM-Random (lr=1e-3, normal init) - Ablation"
    echo "  GPT-04    OpenWebText Baseline (lr=3e-4)"
    echo "  GPT-05    OpenWebText OELM-Freeze (lr=1e-3)"
    echo "  GPT-06    WikiText-103 Baseline (lr=3e-4)"
    echo "  GPT-07    WikiText-103 OELM-Freeze (lr=1e-3)"
    echo ""
    echo "Examples:"
    echo "  $0 GPT-01      # Run TinyStories Baseline"
    echo "  $0 GPT-02      # Run TinyStories OELM-Freeze"
    echo "  $0 all         # Run all experiments sequentially"
}

# 运行单个实验
run_experiment() {
    local exp_id=$1
    local dataset=$2
    local model_type=$3
    local lr=$4
    local max_steps=${5:-$DEFAULT_MAX_STEPS}

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running Experiment: $exp_id${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Dataset: $dataset"
    echo "Model: $model_type"
    echo "Learning Rate: $lr"
    echo "Max Steps: $max_steps"
    echo ""

    # 检查数据是否存在
    local data_path="$DATA_DIR/$dataset/train.bin"
    if [ ! -f "$data_path" ]; then
        echo -e "${RED}Error: Data not found at $data_path${NC}"
        echo "Please prepare data first:"
        echo "  python data/prepare_data.py --dataset $dataset"
        return 1
    fi

    # 设置输出目录
    local out_dir="$OUTPUT_BASE/${exp_id}_${dataset}_${model_type}"
    mkdir -p "$out_dir"

    # 构建命令
    local cmd="python $TRAIN_SCRIPT"
    cmd="$cmd --model_type $model_type"
    cmd="$cmd --dataset $dataset"
    cmd="$cmd --data_path $data_path"
    cmd="$cmd --out_dir $out_dir"
    cmd="$cmd --max_lr $lr"
    cmd="$cmd --d_model $DEFAULT_D_MODEL"
    cmd="$cmd --num_layers $DEFAULT_NUM_LAYERS"
    cmd="$cmd --num_heads $DEFAULT_NUM_HEADS"
    cmd="$cmd --d_ff $DEFAULT_D_FF"
    cmd="$cmd --seq_len $DEFAULT_SEQ_LEN"
    cmd="$cmd --batch_size $DEFAULT_BATCH_SIZE"
    cmd="$cmd --max_steps $max_steps"
    cmd="$cmd --val_interval 1000"
    cmd="$cmd --save_interval 5000"
    cmd="$cmd --use_amp"

    echo "Command: $cmd"
    echo ""

    # 运行训练
    eval $cmd

    echo -e "${GREEN}✓ Experiment $exp_id completed!${NC}"
    echo "Output directory: $out_dir"
    echo "Timing stats: $out_dir/timing_stats.json"
    echo ""
}

# 主逻辑
case "$1" in
    GPT-01)
        run_experiment "GPT-01" "tinystories" "baseline" "3e-4" 100000
        ;;
    GPT-02)
        run_experiment "GPT-02" "tinystories" "oelm_v2" "1e-3" 100000
        ;;
    GPT-03)
        run_experiment "GPT-03" "tinystories" "oelm_random" "1e-3" 100000
        ;;
    GPT-04)
        run_experiment "GPT-04" "openwebtext" "baseline" "3e-4" 150000
        ;;
    GPT-05)
        run_experiment "GPT-05" "openwebtext" "oelm_v2" "1e-3" 150000
        ;;
    GPT-06)
        run_experiment "GPT-06" "wikitext103" "baseline" "3e-4" 200000
        ;;
    GPT-07)
        run_experiment "GPT-07" "wikitext103" "oelm_v2" "1e-3" 200000
        ;;
    all)
        echo -e "${YELLOW}Running all Phase 3 experiments sequentially...${NC}"
        echo "This will take approximately 4-5 days."
        echo ""
        run_experiment "GPT-01" "tinystories" "baseline" "3e-4" 100000
        run_experiment "GPT-02" "tinystories" "oelm_v2" "1e-3" 100000
        run_experiment "GPT-03" "tinystories" "oelm_random" "1e-3" 100000
        run_experiment "GPT-04" "openwebtext" "baseline" "3e-4" 150000
        run_experiment "GPT-05" "openwebtext" "oelm_v2" "1e-3" 150000
        run_experiment "GPT-06" "wikitext103" "baseline" "3e-4" 200000
        run_experiment "GPT-07" "wikitext103" "oelm_v2" "1e-3" 200000
        echo -e "${GREEN}All Phase 3 experiments completed!${NC}"
        ;;
    *)
        show_help
        exit 1
        ;;
esac
