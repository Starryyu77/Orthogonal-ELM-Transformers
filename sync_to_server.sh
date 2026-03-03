#!/bin/bash
# 同步代码到NTU GPU43服务器
# 运行后需要输入密码: GoodLuck2025!zty

cd "$(dirname "$0")"

echo "同步代码到NTU GPU43服务器..."
echo "密码: GoodLuck2025!zty"
echo ""

rsync -avz --progress \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'venv' --exclude '.claude' --exclude 'out' \
    --exclude '*.pt' --exclude '*.bin' --exclude '.DS_Store' \
    --exclude 'data/*.bin' --exclude 'logs' \
    "./" "tianyu016@10.97.216.128:/projects/Orthogonal_ELM_Transformers/"

echo ""
echo "同步完成!"
