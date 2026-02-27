#!/bin/bash
# =============================================================================
# 实验监控脚本
# =============================================================================
# 功能: 监控所有运行中的实验状态
#
# 用法:
#   ./monitor_experiments.sh           # 显示所有实验状态
#   ./monitor_experiments.sh live      # 实时刷新监控
#   ./monitor_experiments.sh log EXP   # 查看特定实验日志
#
# =============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

show_status() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}      实验运行状态监控${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # 检查tmux会话
    echo -e "${YELLOW}运行中的Tmux会话:${NC}"
    tmux ls 2>/dev/null | grep -E "gpt|exp" || echo "  无运行中的实验会话"
    echo ""

    # 检查GPU状态
    echo -e "${YELLOW}GPU状态:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
                   --format=csv,noheader | \
        while IFS=',' read -r idx name util mem_used mem_total; do
            printf "  GPU%s: %-20s 利用率: %4s  显存: %s / %s\n" \
                   "$idx" "$name" "$util" "$mem_used" "$mem_total"
        done
    else
        echo "  nvidia-smi 不可用"
    fi
    echo ""

    # 检查输出目录
    echo -e "${YELLOW}最近更新的输出目录:${NC}"
    find "$PROJECT_DIR/gpt-oelm-project/outputs" -name "training.log" -mtime -1 \
         -exec ls -lh {} \; 2>/dev/null | head -10 || echo "  无近期日志"
}

show_live() {
    while true; do
        clear
        show_status
        echo ""
        echo -e "${YELLOW}按 Ctrl+C 退出实时监控${NC}"
        sleep 5
    done
}

show_log() {
    local exp_id=$1
    local log_file="$PROJECT_DIR/gpt-oelm-project/outputs/${exp_id}_*/training.log"

    if ls $log_file 1> /dev/null 2>&1; then
        echo -e "${GREEN}显示 ${exp_id} 的最新日志 (按 Ctrl+C 退出):${NC}"
        tail -f $log_file
    else
        echo -e "${RED}错误: 找不到 ${exp_id} 的日志文件${NC}"
        echo "可用实验:"
        ls -1 "$PROJECT_DIR/gpt-oelm-project/outputs/" 2>/dev/null || echo "  无输出目录"
    fi
}

# 主逻辑
case "$1" in
    live)
        show_live
        ;;
    log)
        if [ -z "$2" ]; then
            echo "用法: $0 log <experiment_id>"
            echo "例如: $0 log GPT-01"
            exit 1
        fi
        show_log "$2"
        ;;
    *)
        show_status
        echo ""
        echo "用法:"
        echo "  $0           # 显示当前状态"
        echo "  $0 live      # 实时监控"
        echo "  $0 log EXP   # 查看实验日志"
        ;;
esac
