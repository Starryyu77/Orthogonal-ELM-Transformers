#!/bin/bash
# 监控 GPT-01/02 完成状态，自动启动 GPT-03

echo "=========================================="
echo "监控 GPT-01/02 实验状态"
echo "启动 GPT-03 准备就绪"
echo "=========================================="
echo ""

# 检查函数
check_experiment_status() {
    local session=$1
    local name=$2

    if tmux has-session -t "$session" 2>/dev/null; then
        # 获取最后几行输出
        local last_line=$(tmux capture-pane -t "$session" -p 2>/dev/null | tail -1)
        echo "[$name] 运行中: $last_line"
        return 1
    else
        echo "[$name] 会话已结束"
        return 0
    fi
}

# 持续监控
echo "按 Ctrl+C 停止监控"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "实验监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    # 检查 GPT-01
    if check_experiment_status "gpt01_baseline" "GPT-01"; then
        GPT01_DONE=1
    else
        GPT01_DONE=0
    fi

    echo ""

    # 检查 GPT-02
    if check_experiment_status "gpt02_oelm" "GPT-02"; then
        GPT02_DONE=1
    else
        GPT02_DONE=0
    fi

    echo ""
    echo "=========================================="

    # 检查是否可以启动 GPT-03
    if [ "$GPT01_DONE" = "1" ] && [ -z "$GPT03_LAUNCHED" ]; then
        echo ""
        echo "✓ GPT-01 已完成！"
        echo "正在启动 GPT-03 (OELM-Random)..."
        echo ""

        # 启动 GPT-03
        if [ -f "scripts/start_gpt03.sh" ]; then
            ./scripts/start_gpt03.sh
            GPT03_LAUNCHED=1
            echo ""
            echo "✓ GPT-03 已启动!"
            echo "查看: tmux attach -t gpt03_random"
        else
            echo "✗ 启动脚本不存在: scripts/start_gpt03.sh"
        fi
    fi

    if [ "$GPT01_DONE" = "1" ] && [ "$GPT02_DONE" = "1" ]; then
        echo ""
        echo "✓ GPT-01 和 GPT-02 都已完成！"

        if [ -z "$GPT03_LAUNCHED" ]; then
            echo "正在启动 GPT-03..."
            ./scripts/start_gpt03.sh 2>/dev/null || echo "请手动启动: ./scripts/start_gpt03.sh"
        fi

        echo ""
        echo "监控结束。所有实验已完成或已启动。"
        exit 0
    fi

    sleep 30
done
