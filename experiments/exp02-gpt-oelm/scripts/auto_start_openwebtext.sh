#!/bin/bash
# 自动监控 GPU 并在空闲时启动 OpenWebText 实验

echo "=========================================="
echo "OpenWebText 实验自动启动脚本"
echo "=========================================="
echo ""
echo "数据状态: ✅ 已准备 (data/openwebtext/)"
echo "GPU 状态: 检查中..."
echo ""

# 检查 GPU 空闲情况
check_gpu_idle() {
    local gpu_id=$1
    local utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
    if [ "$utilization" -lt 20 ]; then
        return 0  # 空闲
    else
        return 1  # 忙碌
    fi
}

# 等待 GPU 空闲
wait_for_gpu() {
    local target_gpu=$1
    echo "等待 GPU $target_gpu 空闲..."
    while true; do
        if check_gpu_idle $target_gpu; then
            echo "✓ GPU $target_gpu 已空闲!"
            return 0
        fi
        echo "[$(date '+%H:%M:%S')] GPU $target_gpu 仍被占用，60秒后重试..."
        sleep 60
    done
}

# 启动实验
start_experiment() {
    local exp_id=$1
    local gpu_id=$2
    local model_type=$3
    local lr=$4

    export CUDA_VISIBLE_DEVICES=$gpu_id
    local session_name="${exp_id,,}_openwebtext_${model_type//_/}"

    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "会话 $session_name 已存在"
        return 1
    fi

    tmux new-session -d -s "$session_name" -n "training"
    tmux send-keys -t "$session_name" "cd $(pwd)" C-m
    tmux send-keys -t "$session_name" "echo 'Starting $exp_id on GPU $gpu_id'" C-m
    tmux send-keys -t "$session_name" "python3 scripts/train_v2.py \\" C-m
    tmux send-keys -t "$session_name" "  --model_type $model_type \\" C-m
    tmux send-keys -t "$session_name" "  --dataset openwebtext \\" C-m
    tmux send-keys -t "$session_name" "  --max_lr $lr \\" C-m
    tmux send-keys -t "$session_name" "  --max_steps 150000 \\" C-m
    tmux send-keys -t "$session_name" "  --out_dir outputs/${exp_id}_openwebtext_${model_type} \\" C-m
    tmux send-keys -t "$session_name" "  --val_interval 1000 \\" C-m
    tmux send-keys -t "$session_name" "  --use_amp" C-m

    echo "✓ $exp_id 已启动在 GPU $gpu_id (tmux: $session_name)"
}

# 主逻辑
echo "按 Ctrl+C 取消等待"
echo ""

# 等待 GPU 2 启动 GPT-04
wait_for_gpu 2
start_experiment "GPT-04" 2 "baseline" "3e-4"

echo ""

# 等待 GPU 3 启动 GPT-05
wait_for_gpu 3
start_experiment "GPT-05" 3 "oelm_v2" "1e-3"

echo ""
echo "=========================================="
echo "✓ 所有 OpenWebText 实验已启动!"
echo "=========================================="
echo ""
echo "监控命令:"
echo "  tmux attach -t gpt04_openwebtext_baseline"
echo "  tmux attach -t gpt05_openwebtext_oelm"
