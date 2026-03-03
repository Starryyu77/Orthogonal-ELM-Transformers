#!/bin/bash
# 实验监控面板

SERVER="ntu-gpu43"

while true; do
    clear
    echo "=========================================="
    echo "     OELM-FFN 实验监控面板"
    echo "     $(date)"
    echo "=========================================="
    echo ""

    # GPU状态
    echo "【GPU状态】"
    ssh $SERVER "nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null" | while read line; do
        gpu=$(echo $line | cut -d',' -f1 | tr -d ' ')
        temp=$(echo $line | cut -d',' -f2 | tr -d ' ')
        util=$(echo $line | cut -d',' -f3 | tr -d ' ')
        mem_used=$(echo $line | cut -d',' -f4 | tr -d ' ')
        mem_total=$(echo $line | cut -d',' -f5 | tr -d ' ')

        # 进度条
        bar_len=20
        filled=$((util * bar_len / 100))
        empty=$((bar_len - filled))
        bar=$(printf '%*s' "$filled" | tr ' ' '█')
        bar+=$(printf '%*s' "$empty" | tr ' ' '░')

        printf "  GPU %d: %s %3d%% | Temp: %2d°C | Mem: %d/%d MB\n" $gpu "$bar" $util $temp $mem_used $mem_total
    done

    echo ""
    echo "【训练进度】"

    # 检查每个实验
    for log in /tmp/imdb_qk_ffn_1e3.log /tmp/imdb_qk_ffn_3e3.log /tmp/imdb_ffn_only_5e4.log; do
        name=$(basename $log)
        status=$(ssh $SERVER "tail -5 $log 2>/dev/null | grep -E 'Epoch|完成' | tail -1")
        if [ -n "$status" ]; then
            echo "  $name: $status"
        else
            echo "  $name: 等待中..."
        fi
    done

    echo ""
    echo "【快捷命令】"
    echo "  查看详细日志: ssh ntu-gpu43 'tail -100 /tmp/imdb_qk_ffn_1e3.log'"
    echo "  杀死所有实验: ssh ntu-gpu43 'pkill -f train_classification.py'"
    echo ""
    echo "按 Ctrl+C 退出监控"
    echo "下次刷新: 30秒后"

    sleep 30
done
