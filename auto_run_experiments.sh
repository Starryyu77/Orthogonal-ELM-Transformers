#!/bin/bash
# 自动运行实验脚本
# 监控当前实验并在完成后启动新实验

SERVER="ntu-gpu43"
PROJECT_DIR="$HOME/Orthogonal_ELM_Transformers/Train/experiments/phase4-gpt-classification/scripts"

echo "=========================================="
echo "自动实验运行脚本"
echo "开始时间: $(date)"
echo "=========================================="

# 检查实验是否完成的函数
check_experiment_done() {
    local log_file=$1
    ssh $SERVER "grep -q '实验完成' $log_file 2>/dev/null" && return 0 || return 1
}

# 等待实验完成
wait_for_experiments() {
    echo "等待当前实验完成..."
    while true; do
        local all_done=true

        if ! check_experiment_done "/tmp/imdb_qk_ffn_1e3.log"; then
            all_done=false
            echo "  - IMDB QK-FFN lr=1e-3: 运行中..."
        else
            echo "  - IMDB QK-FFN lr=1e-3: 完成 ✓"
        fi

        if ! check_experiment_done "/tmp/imdb_qk_ffn_3e3.log"; then
            all_done=false
            echo "  - IMDB QK-FFN lr=3e-3: 运行中..."
        else
            echo "  - IMDB QK-FFN lr=3e-3: 完成 ✓"
        fi

        if ! check_experiment_done "/tmp/imdb_ffn_only_5e4.log"; then
            all_done=false
            echo "  - IMDB FFN-only lr=5e-4: 运行中..."
        else
            echo "  - IMDB FFN-only lr=5e-4: 完成 ✓"
        fi

        if $all_done; then
            echo "所有实验完成！"
            return 0
        fi

        echo "等待60秒后重新检查..."
        sleep 60
        echo ""
    done
}

# 启动下一批实验
run_next_batch() {
    echo ""
    echo "=========================================="
    echo "启动下一批实验"
    echo "=========================================="

    ssh $SERVER "cd $PROJECT_DIR && nohup ./run_imdb_oelm_qk_ffn.sh 1 2 > /tmp/imdb_qk_ffn_1e2.log 2>&1 &"
    echo "启动: IMDB OELM-QK-FFN lr=1e-2 on GPU 1"

    ssh $SERVER "cd $PROJECT_DIR && nohup ./run_imdb_oelm_ffn_only.sh 2 1 > /tmp/imdb_ffn_only_1e3.log 2>&1 &"
    echo "启动: IMDB OELM-FFN-only lr=1e-3 on GPU 2"

    ssh $SERVER "cd $PROJECT_DIR && nohup ./run_imdb_oelm_ffn_only.sh 3 2 > /tmp/imdb_ffn_only_3e3.log 2>&1 &"
    echo "启动: IMDB OELM-FFN-only lr=3e-3 on GPU 3"
}

# 主流程
wait_for_experiments
run_next_batch

echo ""
echo "=========================================="
echo "第二批实验已启动"
echo "结束时间: $(date)"
echo "=========================================="
