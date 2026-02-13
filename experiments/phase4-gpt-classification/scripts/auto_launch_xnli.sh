#!/bin/bash
# AGNews完成后自动启动XNLI实验
# 此脚本会监控AGNews实验状态，完成后自动启动XNLI

SERVER="ntu-gpu43"
PROJECT_DIR="~/Orthogonal_ELM_Transformers/Train"

echo "=========================================="
echo "自动监控: AGNews -> XNLI"
echo "启动时间: $(date)"
echo "=========================================="

# 检查AGNews是否完成的函数
check_agnews_complete() {
    local result=$(ssh $SERVER "cat $PROJECT_DIR/outputs/AGNews_baseline/results.json 2>/dev/null | grep -c 'best_accuracy'")
    echo $result
}

# 等待AGNews完成
echo "等待 AGNews 实验完成..."
while true; do
    baseline_done=$(check_agnews_complete)

    if [ "$baseline_done" == "1" ]; then
        echo ""
        echo "AGNews 实验已完成！"
        echo "$(date): 准备启动 XNLI 实验..."
        break
    fi

    echo -n "."
    sleep 60  # 每分钟检查一次
done

# 启动XNLI实验
echo ""
echo "=========================================="
echo "启动 XNLI 实验"
echo "=========================================="

ssh $SERVER "
cd $PROJECT_DIR
source ~/projects/oelm/venv/bin/activate

mkdir -p outputs/XNLI_baseline outputs/XNLI_oelm_freeze

echo '启动 XNLI Baseline (GPU 0)...'
tmux new-session -d -s xnli_baseline \"bash experiments/phase4-gpt-classification/scripts/run_xnli_baseline.sh 0 2>&1 | tee outputs/XNLI_baseline/console.log\"

echo '启动 XNLI OELM-Freeze (GPU 1)...'
tmux new-session -d -s xnli_oelm \"bash experiments/phase4-gpt-classification/scripts/run_xnli_oelm.sh 1 2>&1 | tee outputs/XNLI_oelm_freeze/console.log\"

echo ''
echo 'XNLI实验已启动!'
tmux ls | grep xnli
"

echo ""
echo "=========================================="
echo "XNLI 实验已自动启动！"
echo "完成时间: $(date)"
echo "=========================================="
echo ""
echo "监控命令:"
echo "  ssh ntu-gpu43 'tail -f ~/Orthogonal_ELM_Transformers/Train/outputs/XNLI_baseline/console.log'"
echo "  ssh ntu-gpu43 'tail -f ~/Orthogonal_ELM_Transformers/Train/outputs/XNLI_oelm_freeze/console.log'"
