#!/bin/bash
# =============================================================================
# NTU GPU 服务器远程运行脚本
# 无需在服务器上安装 Claude Code CLI
# =============================================================================

SERVER="ntu-gpu43"
PROJECT_DIR="~/projects/oelm"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 构建远程命令
REMOTE_CMD="cd $PROJECT_DIR && source venv/bin/activate"

# 设置 CUDA 使用空闲 GPU (GPU 2,3)
export CUDA_VISIBLE_DEVICES="2,3"

show_help() {
    echo "NTU GPU 服务器远程运行工具"
    echo ""
    echo "用法: $0 <命令> [参数]"
    echo ""
    echo "可用命令:"
    echo "  train oelm|gpt [config]     训练模型"
    echo "  train-multi oelm|gpt [gpus] 多 GPU 训练"
    echo "  benchmark [size]            运行基准测试"
    echo "  check                       检查环境"
    echo "  python <args>               运行 Python"
    echo "  bash                        交互式 Bash"
    echo "  jupyter                     启动 Jupyter Lab"
    echo "  tensorboard                 启动 TensorBoard"
    echo "  sync                        同步代码到服务器"
    echo "  status                      查看 GPU 状态"
    echo "  kill-all                    杀死所有 Python 进程 (谨慎使用)"
    echo ""
    echo "示例:"
    echo "  $0 train oelm small         训练 OELM small 模型"
    echo "  $0 benchmark small          基准测试"
    echo "  $0 python train.py --help   查看 train.py 帮助"
}

case "$1" in
    train)
        MODEL_TYPE=${2:-oelm}
        CONFIG=${3:-small}
        echo -e "${BLUE}启动训练: $MODEL_TYPE ($CONFIG)${NC}"
        echo -e "${YELLOW}使用 GPU 2,3 (避免显存冲突)${NC}"
        ssh $SERVER "cd $PROJECT_DIR && source venv/bin/activate && export CUDA_VISIBLE_DEVICES=2 && python train.py --model_type $MODEL_TYPE --data_path data/tinystories/train.bin --out_dir out/${MODEL_TYPE}_${CONFIG} --batch_size 8 --max_steps 10000"
        ;;

    train-multi)
        MODEL_TYPE=${2:-oelm}
        NUM_GPUS=${3:-2}
        echo -e "${BLUE}启动多 GPU 训练: $MODEL_TYPE (GPUs: $NUM_GPUS)${NC}"
        echo -e "${YELLOW}使用 GPU 2,3${NC}"
        ssh $SERVER "cd $PROJECT_DIR && source venv/bin/activate && export CUDA_VISIBLE_DEVICES=2,3 && torchrun --standalone --nproc_per_node=$NUM_GPUS train.py --model_type $MODEL_TYPE --data_path data/tinystories/train.bin --out_dir out/${MODEL_TYPE}_multi --batch_size 4 --max_steps 10000"
        ;;

    benchmark)
        SIZE=${2:-small}
        echo -e "${BLUE}运行基准测试 ($SIZE)${NC}"
        echo -e "${YELLOW}使用 GPU 2 (避免显存冲突)${NC}"
        # 修改 benchmark.py 使用较小的 batch size
        ssh $SERVER "cd $PROJECT_DIR && source venv/bin/activate && export CUDA_VISIBLE_DEVICES=2 && python -c \"
import sys
sys.path.insert(0, '.')
from benchmark import *
import argparse

args = argparse.Namespace()
args.model_type = 'oelm'
args.model_size = '$SIZE'
args.vocab_size = 50257
args.batch_size = 4
args.seq_len = 512
args.num_steps = 50
args.warmup_steps = 5
args.skip_training = False
args.skip_inference = False
args.memory_scaling = False
args.compare = True

compare_models(args)
\""
        ;;

    check)
        echo -e "${BLUE}检查环境${NC}"
        ssh $SERVER "$REMOTE_CMD && python check_env.py"
        ;;

    python)
        shift
        echo -e "${BLUE}运行: python $@${NC}"
        ssh $SERVER "$REMOTE_CMD && export CUDA_VISIBLE_DEVICES=2 && python $@"
        ;;

    bash)
        echo -e "${BLUE}连接到服务器 Bash...${NC}"
        ssh -t $SERVER "$REMOTE_CMD && export CUDA_VISIBLE_DEVICES=2,3 && bash"
        ;;

    jupyter)
        echo -e "${BLUE}启动 Jupyter Lab (端口 8888)...${NC}"
        echo "在浏览器中打开: http://localhost:8888"
        ssh -L 8888:localhost:8888 -t $SERVER "$REMOTE_CMD && jupyter lab --ip=0.0.0.0 --no-browser --port=8888"
        ;;

    tensorboard)
        echo -e "${BLUE}启动 TensorBoard (端口 6006)...${NC}"
        echo "在浏览器中打开: http://localhost:6006"
        ssh -L 6006:localhost:6006 -t $SERVER "$REMOTE_CMD && tensorboard --logdir=out --port=6006"
        ;;

    sync)
        echo -e "${BLUE}同步代码到服务器...${NC}"
        ntu-sync 2>/dev/null || {
            rsync -avz --progress \
                --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
                --exclude 'venv' --exclude '.claude' --exclude 'out' \
                --exclude '*.pt' --exclude '*.bin' --exclude '.DS_Store' \
                --exclude 'data/*.bin' \
                "." "$SERVER:~/projects/oelm/"
        }
        ;;

    status)
        echo -e "${BLUE}GPU 状态:${NC}"
        ssh $SERVER "nvidia-smi"
        ;;

    kill-all)
        echo -e "${RED}警告: 这将杀死所有 Python 进程${NC}"
        read -p "确定继续? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ssh $SERVER "pkill -f python"
            echo -e "${GREEN}已杀死所有 Python 进程${NC}"
        fi
        ;;

    *)
        show_help
        ;;
esac
