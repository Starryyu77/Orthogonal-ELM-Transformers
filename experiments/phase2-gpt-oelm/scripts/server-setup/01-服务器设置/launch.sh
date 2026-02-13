#!/bin/bash
# =============================================================================
# 正交随机注意力训练启动脚本
# =============================================================================
# 使用方法:
#   chmod +x launch.sh
#   ./launch.sh [single|multi|slurm] [config_name]
#
# 示例:
#   ./launch.sh single tiny          # 单GPU，tiny配置
#   ./launch.sh multi medium         # 多GPU，medium配置
#   ./launch.sh slurm large          # Slurm集群，large配置
# =============================================================================

set -e

# 默认参数
MODE=${1:-single}
CONFIG=${2:-tiny}
DATASET=${3:-tinystories}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  正交随机注意力训练启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Mode: $MODE"
echo "Config: $CONFIG"
echo "Dataset: $DATASET"
echo ""

# 检查环境
check_environment() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found${NC}"
        exit 1
    fi
    
    if ! python -c "import torch" 2>/dev/null; then
        echo -e "${RED}Error: PyTorch not installed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Environment check passed${NC}"
}

# 准备数据
prepare_data() {
    if [ ! -f "data/${DATASET}_train.bin" ]; then
        echo -e "${YELLOW}Data not found, preparing...${NC}"
        cd data
        python prepare_data.py --dataset $DATASET --output_dir .
        cd ..
    else
        echo -e "${GREEN}Data already prepared${NC}"
    fi
}

# 获取配置参数
get_config_args() {
    case $CONFIG in
        tiny)
            echo "--d_model 128 --n_layers 4 --n_heads 4 --d_head 32 --d_ff 512 --max_seq_len 256 --num_orthogonal_features 64 --batch_size 128 --max_iters 5000 --eval_interval 500"
            ;;
        small)
            echo "--d_model 384 --n_layers 6 --n_heads 6 --d_head 64 --d_ff 1536 --max_seq_len 512 --num_orthogonal_features 192 --batch_size 64 --max_iters 50000 --eval_interval 500"
            ;;
        medium)
            echo "--d_model 768 --n_layers 12 --n_heads 12 --d_head 64 --d_ff 3072 --max_seq_len 512 --num_orthogonal_features 384 --batch_size 32 --max_iters 100000 --eval_interval 1000"
            ;;
        large)
            echo "--d_model 1024 --n_layers 24 --n_heads 16 --d_head 64 --d_ff 4096 --max_seq_len 1024 --num_orthogonal_features 512 --batch_size 16 --max_iters 200000 --eval_interval 2000"
            ;;
        standard)
            echo "--d_model 768 --n_layers 12 --n_heads 12 --d_head 64 --d_ff 3072 --max_seq_len 512 --no_orthogonal_attention --batch_size 32 --max_iters 100000 --eval_interval 1000"
            ;;
        *)
            echo ""
            ;;
    esac
}

# 单GPU训练
run_single() {
    echo -e "${GREEN}Starting single GPU training...${NC}"
    
    CONFIG_ARGS=$(get_config_args)
    
    python train.py \
        --dataset $DATASET \
        $CONFIG_ARGS \
        --data_dir data \
        --out_dir out/$CONFIG \
        --wandb_run_name "${CONFIG}_${DATASET}_single"
}

# 多GPU训练
run_multi() {
    NUM_GPUS=${4:-4}
    echo -e "${GREEN}Starting multi-GPU training with $NUM_GPUS GPUs...${NC}"
    
    CONFIG_ARGS=$(get_config_args)
    
    # 调整batch size
    if [ "$CONFIG" = "medium" ]; then
        CONFIG_ARGS="$CONFIG_ARGS --batch_size 8"
    elif [ "$CONFIG" = "large" ]; then
        CONFIG_ARGS="$CONFIG_ARGS --batch_size 4"
    fi
    
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        train.py \
        --dataset $DATASET \
        $CONFIG_ARGS \
        --data_dir data \
        --out_dir out/$CONFIG \
        --wandb_run_name "${CONFIG}_${DATASET}_multi${NUM_GPUS}"
}

# Slurm训练
run_slurm() {
    echo -e "${GREEN}Starting Slurm training...${NC}"
    
    # 创建Slurm脚本
    cat > slurm_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=orthogonal-attention
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm-%j.out

CONFIG=$1
DATASET=$2

# 加载环境
module load cuda/11.8
source activate pytorch

# 获取节点列表
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node ($head_node_ip)"
echo "Nodes: ${nodes[@]}"

# 启动训练
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    train.py \
    --dataset $DATASET \
    --d_model 1024 --n_layers 24 --n_heads 16 --d_head 64 \
    --d_ff 4096 --max_seq_len 1024 --num_orthogonal_features 512 \
    --batch_size 4 --max_iters 200000 --eval_interval 2000 \
    --data_dir data --out_dir out/$CONFIG \
    --wandb_run_name "${CONFIG}_${DATASET}_slurm"
EOF
    
    chmod +x slurm_job.sh
    sbatch slurm_job.sh $CONFIG $DATASET
    
    echo -e "${GREEN}Slurm job submitted${NC}"
}

# 恢复训练
run_resume() {
    CHECKPOINT=${4:-"out/latest.pt"}
    echo -e "${GREEN}Resuming training from $CHECKPOINT...${NC}"
    
    CONFIG_ARGS=$(get_config_args)
    
    python train.py \
        --dataset $DATASET \
        $CONFIG_ARGS \
        --data_dir data \
        --out_dir out/$CONFIG \
        --resume $CHECKPOINT \
        --wandb_run_name "${CONFIG}_${DATASET}_resumed"
}

# 主函数
main() {
    check_environment
    prepare_data
    
    case $MODE in
        single)
            run_single
            ;;
        multi)
            run_multi
            ;;
        slurm)
            run_slurm
            ;;
        resume)
            run_resume
            ;;
        *)
            echo -e "${RED}Unknown mode: $MODE${NC}"
            echo "Usage: $0 [single|multi|slurm|resume] [config_name] [dataset] [extra_arg]"
            echo ""
            echo "Examples:"
            echo "  $0 single tiny tinystories"
            echo "  $0 multi medium tinystories 4"
            echo "  $0 slurm large openwebtext"
            echo "  $0 resume tiny tinystories out/latest.pt"
            exit 1
            ;;
    esac
}

main
