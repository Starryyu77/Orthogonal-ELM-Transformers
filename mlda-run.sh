#!/bin/bash
# =============================================================================
# MLDA GPU 服务器远程运行脚本
# 用于控制 MLDA GPU 上的 OELM 训练
# =============================================================================

SERVER="s125mdg43_10@gpu43.dynip.ntu.edu.sg"
PROJECT_DIR="~/Orthogonal_ELM_Transformers/Train"
VENV_PATH="~/projects/oelm/venv/bin/activate"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

show_help() {
    echo "MLDA GPU 远程运行工具"
    echo ""
    echo "用法: $0 <命令> [参数]"
    echo ""
    echo "可用命令:"
    echo "  train-gpt                   启动 GPT Medium-512 训练"
    echo "  train-oelm                  启动 OELM Medium-512 训练"
    echo "  train-both                  同时启动 GPT + OELM 训练"
    echo "  status                      查看 GPU 状态"
    echo "  logs                        查看训练日志"
    echo "  kill-all                    杀死所有训练进程"
    echo "  sync                        同步代码到服务器"
    echo "  bash                        交互式 Bash"
    echo ""
    echo "示例:"
    echo "  $0 train-gpt                训练 GPT 基线"
    echo "  $0 train-oelm               训练 OELM"
    echo "  $0 train-both               同时训练两个模型"
    echo "  $0 status                   查看 GPU 状态"
}

case "$1" in
    train-gpt)
        echo -e "${BLUE}启动 GPT Medium-512 训练...${NC}"
        TRAIN_SCRIPT="scripts/02-训练脚本/train.py"
        ssh $SERVER "cd $PROJECT_DIR && mkdir -p logs && source $VENV_PATH && tmux new-session -d -s gpt_train \"cd $PROJECT_DIR && source $VENV_PATH && export PYTHONPATH=$PROJECT_DIR:\$PYTHONPATH && python -m torch.distributed.run --nproc_per_node=2 --master_port=29500 \\\"$TRAIN_SCRIPT\\\" --model_type gpt --d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 --seq_len 512 --batch_size 8 --max_steps 100000 --data_path data/tiny_stories/train.bin --out_dir models/checkpoints/gpt_medium512 2>&1 | tee logs/gpt_train.log\""
        echo -e "${GREEN}✓ GPT 训练已在 tmux 会话 'gpt_train' 中启动${NC}"
        echo -e "${YELLOW}查看日志: ssh $SERVER 'tail -f $PROJECT_DIR/logs/gpt_train.log'${NC}"
        ;;

    train-oelm)
        echo -e "${BLUE}启动 OELM Medium-512 训练...${NC}"
        TRAIN_SCRIPT="scripts/02-训练脚本/train.py"
        ssh $SERVER "cd $PROJECT_DIR && mkdir -p logs && source $VENV_PATH && tmux new-session -d -s oelm_train \"cd $PROJECT_DIR && source $VENV_PATH && export PYTHONPATH=$PROJECT_DIR:\$PYTHONPATH && export CUDA_VISIBLE_DEVICES=2,3 && python -m torch.distributed.run --nproc_per_node=2 --master_port=29501 \\\"$TRAIN_SCRIPT\\\" --model_type oelm --d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 --seq_len 512 --batch_size 8 --max_steps 100000 --data_path data/tiny_stories/train.bin --out_dir models/checkpoints/oelm_medium512 2>&1 | tee logs/oelm_train.log\""
        echo -e "${GREEN}✓ OELM 训练已在 tmux 会话 'oelm_train' 中启动${NC}"
        echo -e "${YELLOW}查看日志: ssh $SERVER 'tail -f $PROJECT_DIR/logs/oelm_train.log'${NC}"
        ;;

    train-both)
        echo -e "${BLUE}同时启动 GPT + OELM 训练...${NC}"
        $0 train-gpt
        $0 train-oelm
        echo ""
        echo -e "${GREEN}两个训练任务已启动！${NC}"
        echo -e "查看状态: $0 status"
        echo -e "查看日志: $0 logs"
        ;;

    status)
        echo -e "${BLUE}GPU 状态:${NC}"
        ssh $SERVER "nvidia-smi"
        echo ""
        echo -e "${BLUE}运行中的 tmux 会话:${NC}"
        ssh $SERVER "tmux ls 2>/dev/null || echo '没有运行中的会话'"
        ;;

    logs)
        echo -e "${BLUE}训练日志:${NC}"
        echo "GPT 日志:"
        ssh $SERVER "tail -20 $PROJECT_DIR/logs/gpt_train.log 2>/dev/null || echo '日志不存在'"
        echo ""
        echo "OELM 日志:"
        ssh $SERVER "tail -20 $PROJECT_DIR/logs/oelm_train.log 2>/dev/null || echo '日志不存在'"
        ;;

    kill-all)
        echo -e "${RED}警告: 这将杀死所有 Python 和 tmux 会话${NC}"
        read -p "确定继续? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ssh $SERVER "tmux kill-session -t gpt_train 2>/dev/null; tmux kill-session -t oelm_train 2>/dev/null; pkill -f 'train.py'"
            echo -e "${GREEN}已停止所有训练${NC}"
        fi
        ;;

    sync)
        echo -e "${BLUE}同步代码到服务器...${NC}"
        rsync -avz --progress \
            --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
            --exclude 'venv' --exclude '.claude' --exclude 'out' \
            --exclude '*.pt' --exclude '*.bin' --exclude '.DS_Store' \
            --exclude 'data/*.bin' --exclude 'logs' \
            "./" "$SERVER:$PROJECT_DIR/"
        echo -e "${GREEN}同步完成${NC}"
        ;;

    bash)
        echo -e "${BLUE}连接到 MLDA GPU Bash...${NC}"
        ssh -t $SERVER "cd $PROJECT_DIR && source $VENV_PATH && bash"
        ;;

    *)
        show_help
        ;;
esac
