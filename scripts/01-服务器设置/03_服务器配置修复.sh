#!/bin/bash
# =============================================================================
# OELM 项目服务器环境配置 (修复版 - 无需 sudo)
# =============================================================================

set -e

echo "========================================"
echo "  OELM 项目服务器环境配置 (修复版)"
echo "========================================"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# 检查系统环境
# =============================================================================
echo -e "\n${BLUE}[1/6] 检查系统环境...${NC}"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 未安装${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}  Python 版本: $PYTHON_VERSION${NC}"

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}  CUDA 可用:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}  警告: 未检测到 CUDA${NC}"
    GPU_AVAILABLE=false
fi

# 检查是否有 conda
if command -v conda &> /dev/null; then
    echo -e "${GREEN}  Conda 可用${NC}"
    HAS_CONDA=true
else
    HAS_CONDA=false
fi

# =============================================================================
# 创建项目目录
# =============================================================================
echo -e "\n${BLUE}[2/6] 创建项目目录...${NC}"

PROJECT_DIR="$HOME/projects/oelm"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
echo -e "${GREEN}  项目目录: $PROJECT_DIR${NC}"

# =============================================================================
# 创建虚拟环境 (使用可用方法)
# =============================================================================
echo -e "\n${BLUE}[3/6] 创建虚拟环境...${NC}"

VENV_DIR="$PROJECT_DIR/venv"

# 方法1: 尝试使用 virtualenv
if command -v virtualenv &> /dev/null; then
    echo -e "${GREEN}  使用 virtualenv 创建环境${NC}"
    virtualenv "$VENV_DIR"
# 方法2: 使用 conda
elif [ "$HAS_CONDA" = true ]; then
    echo -e "${GREEN}  使用 Conda 创建环境${NC}"
    conda create -p "$VENV_DIR" python=3.8 -y
# 方法3: 使用 python -m venv (尝试安装 ensurepip)
else
    echo -e "${YELLOW}  尝试使用 python3-venv${NC}"
    # 尝试从已安装的包中获取 venv
    python3 -m pip install --user virtualenv 2>/dev/null || true

    if command -v virtualenv &> /dev/null || [ -f "$HOME/.local/bin/virtualenv" ]; then
        "$HOME/.local/bin/virtualenv" "$VENV_DIR" 2>/dev/null || virtualenv "$VENV_DIR"
    else
        # 最后手段：使用用户级 pip (不推荐但可用)
        echo -e "${YELLOW}  警告: 无法创建虚拟环境，将使用用户级 pip${NC}"
        VENV_DIR=""
    fi
fi

# 激活虚拟环境
if [ -n "$VENV_DIR" ] && [ -d "$VENV_DIR" ]; then
    if [ "$HAS_CONDA" = true ] && [ -d "$VENV_DIR/bin/conda" ]; then
        source activate "$VENV_DIR"
    else
        source "$VENV_DIR/bin/activate"
    fi
    echo -e "${GREEN}  虚拟环境已激活${NC}"
else
    echo -e "${YELLOW}  使用用户级 Python 环境${NC}"
    # 确保 pip 是最新的
    python3 -m pip install --user --upgrade pip
fi

# =============================================================================
# 安装 PyTorch 和 CUDA
# =============================================================================
echo -e "\n${BLUE}[4/6] 安装 PyTorch (带 CUDA 支持)...${NC}"

if [ "$GPU_AVAILABLE" = true ]; then
    # 检测 CUDA 版本
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/')
    echo -e "${GREEN}  检测到 CUDA 版本: $CUDA_VERSION${NC}"

    # 安装 PyTorch (使用 CUDA 11.8，兼容性好)
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${YELLOW}  安装 CPU 版本 PyTorch${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}  PyTorch 安装完成${NC}"
python3 -c "import torch; print(f'  PyTorch 版本: {torch.__version__}'); print(f'  CUDA 可用: {torch.cuda.is_available()}')"

# =============================================================================
# 安装项目依赖
# =============================================================================
echo -e "\n${BLUE}[5/6] 安装项目依赖...${NC}"

pip install --upgrade pip
pip install numpy tqdm datasets tiktoken wandb matplotlib seaborn pytest black flake8

echo -e "${GREEN}  依赖安装完成${NC}"

# =============================================================================
# 创建辅助脚本
# =============================================================================
echo -e "\n${BLUE}[6/6] 创建辅助脚本...${NC}"

# 创建快速启动脚本
cat > "$PROJECT_DIR/start_claude.sh" << 'EOF'
#!/bin/bash
# 启动 Claude Code（自动激活虚拟环境）
cd "$HOME/projects/oelm"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -d "venv" ]; then
    source activate venv
fi
claude "$@"
EOF
chmod +x "$PROJECT_DIR/start_claude.sh"

# 创建环境检查脚本
cat > "$PROJECT_DIR/check_env.py" << 'EOF'
#!/usr/bin/env python3
"""检查环境配置"""

import sys
import torch
import numpy as np

def check_environment():
    print("=" * 50)
    print("环境检查报告")
    print("=" * 50)

    # Python 版本
    print(f"\nPython 版本: {sys.version}")

    # PyTorch
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    显存: {props.total_memory / 1024**3:.1f} GB")
            print(f"    计算能力: {props.major}.{props.minor}")

        # 测试 GPU 计算
        print("\n测试 GPU 计算...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x.t())
        print(f"  GPU 计算测试: 通过 ✓")

    print("\n" + "=" * 50)
    print("环境检查完成")
    print("=" * 50)

if __name__ == "__main__":
    check_environment()
EOF
chmod +x "$PROJECT_DIR/check_env.py"

# 创建训练快捷脚本
cat > "$PROJECT_DIR/train_oelm.sh" << 'EOF'
#!/bin/bash
# 训练 OELM 模型的快捷脚本

cd "$HOME/projects/oelm"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -d "venv" ]; then
    source activate venv
fi

# 默认配置
MODEL_TYPE=${1:-oelm}
CONFIG=${2:-small}
DATASET=${3:-tinystories}

echo "开始训练: model=$MODEL_TYPE, config=$CONFIG, dataset=$DATASET"

python train.py \
    --model_type $MODEL_TYPE \
    --data_path data/$DATASET/train.bin \
    --out_dir out/${MODEL_TYPE}_${CONFIG} \
    --batch_size 32 \
    --max_steps 100000 \
    --use_wandb
EOF
chmod +x "$PROJECT_DIR/train_oelm.sh"

# 创建多 GPU 训练脚本
cat > "$PROJECT_DIR/train_multi_gpu.sh" << 'EOF'
#!/bin/bash
# 多 GPU 训练脚本

cd "$HOME/projects/oelm"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -d "venv" ]; then
    source activate venv
fi

MODEL_TYPE=${1:-oelm}
NUM_GPUS=${2:-4}

echo "开始多 GPU 训练: model=$MODEL_TYPE, gpus=$NUM_GPUS"

torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    train.py \
    --model_type $MODEL_TYPE \
    --data_path data/tinystories/train.bin \
    --out_dir out/${MODEL_TYPE}_multi \
    --batch_size 8 \
    --max_steps 100000 \
    --use_wandb
EOF
chmod +x "$PROJECT_DIR/train_multi_gpu.sh"

echo -e "${GREEN}  辅助脚本已创建:${NC}"
echo "    - start_claude.sh: 启动 Claude Code"
echo "    - check_env.py: 检查环境"
echo "    - train_oelm.sh: 快速训练"
echo "    - train_multi_gpu.sh: 多 GPU 训练"

# =============================================================================
# 总结
# =============================================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  服务器环境配置完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "项目目录: $PROJECT_DIR"
echo "虚拟环境: ${VENV_DIR:-"使用系统 Python"}"
echo ""
echo "常用命令:"
echo "  检查环境:           python check_env.py"
echo "  训练模型:           ./train_oelm.sh oelm small tinystories"
echo "  多 GPU 训练:        ./train_multi_gpu.sh oelm 4"
echo "  运行基准测试:       python benchmark.py --compare"
echo ""
echo "Claude Code 已准备好使用！"
echo "  ./start_claude.sh"
