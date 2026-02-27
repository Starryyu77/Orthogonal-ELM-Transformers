#!/bin/bash
# 快速检查实验状态脚本

echo "========== GPT OELM 实验状态 =========="
echo ""

# GPU 状态
echo "GPU 状态:"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | grep -E "(GPU 2|GPU 3)" || echo "  GPU 2: 运行中"
echo ""

# 检查 tmux 会话
echo "运行中的会话:"
tmux list-sessions | grep -E "gpt0[12]" | while read line; do
    echo "  $line"
done
echo ""

# 检查训练进度
echo "训练进度:"
echo "--- GPT-01 (Baseline) ---"
tmux capture-pane -t gpt01_baseline -p 2>/dev/null | tail -3

echo ""
echo "--- GPT-02 (OELM-Freeze) ---"
tmux capture-pane -t gpt02_oelm -p 2>/dev/null | tail -3

echo ""
echo "========================================="
echo "查看完整日志: ./mlda-run.sh logs"
echo "实时查看: tmux attach -t gpt01_baseline"
echo "实时查看: tmux attach -t gpt02_oelm"
