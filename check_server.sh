#!/bin/bash
# 检查服务器实验状态的脚本

echo "=========================================="
echo "检查NTU GPU43服务器实验状态"
echo "时间: $(date)"
echo "=========================================="
echo ""

# 检查连接
if ! ssh ntu-gpu43 "echo 'Connected'" 2>/dev/null; then
    echo "❌ 无法连接到服务器，请稍后重试"
    exit 1
fi

echo "✅ 服务器连接正常"
echo ""

# GPU状态
echo "【GPU状态】"
ssh ntu-gpu43 "nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader" 2>/dev/null | while IFS=',' read -r idx temp util mem; do
    printf "  GPU %s: %3s%% | Temp: %2s°C | Mem: %5s MiB\n" "$idx" "$util" "$temp" "$mem"
done
echo ""

# 检查实验进度
echo "【实验进度】"
for log in /tmp/imdb_qk_ffn_1e3.log /tmp/imdb_qk_ffn_3e3.log /tmp/imdb_qk_ffn_1e2.log \
           /tmp/imdb_ffn_only_5e4.log /tmp/imdb_ffn_only_1e3.log \
           /tmp/agnews_qk_ffn_1e3.log /tmp/agnews_qk_ffn_3e3.log /tmp/agnews_ffn_only_5e4.log; do
    if ssh ntu-gpu43 "test -f $log" 2>/dev/null; then
        name=$(basename $log .log)
        status=$(ssh ntu-gpu43 "tail -1 $log 2>/dev/null | grep -E 'Epoch|完成' | head -1")
        if [ -n "$status" ]; then
            echo "  $name: $status"
        else
            echo "  $name: 运行中..."
        fi
    fi
done
echo ""

# 检查结果文件
echo "【已完成实验】"
ssh ntu-gpu43 "ls ~/Orthogonal_ELM_Transformers/Train/outputs_phase4/*/results.json 2>/dev/null" | while read f; do
    dir=$(dirname "$f")
    name=$(basename "$dir")
    acc=$(ssh ntu-gpu43 "cat $f 2>/dev/null | grep best_accuracy | head -1")
    echo "  $name: $acc"
done

echo ""
echo "=========================================="
