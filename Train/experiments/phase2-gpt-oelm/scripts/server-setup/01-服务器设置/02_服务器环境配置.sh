#!/bin/bash
# =============================================================================
# GPU 服务器环境设置脚本
# 用于配置 OELM 项目的远程开发环境
# =============================================================================

set -e

echo "========================================"
echo "  OELM 项目服务器环境配置"
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
echo -e "\n${BLUE}[1/7] 检查系统环境...${NC}"

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
else
    echo -e "${YELLOW}  警告: 未检测到 CUDA，将使用 CPU 模式${NC}"
fi

# =============================================================================
# 创建项目目录
# =============================================================================
echo -e "\n${BLUE}[2/7] 创建项目目录...${NC}"

PROJECT_DIR="$HOME/projects/oelm"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
echo -e "${GREEN}  项目目录: $PROJECT_DIR${NC}"

# =============================================================================
# 创建 Python 虚拟环境
# =============================================================================
echo -e "\n${BLUE}[3/7] 创建 Python 虚拟环境...${NC}"

VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}  虚拟环境已创建: $VENV_DIR${NC}"
else
    echo -e "${YELLOW}  虚拟环境已存在，跳过创建${NC}"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}  虚拟环境已激活${NC}"

# =============================================================================
# 安装 PyTorch 和 CUDA
# =============================================================================
echo -e "\n${BLUE}[4/7] 安装 PyTorch (带 CUDA 支持)...${NC}"

# 检测 CUDA 版本并安装对应 PyTorch
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo -e "${GREEN}  检测到 CUDA 版本: $CUDA_VERSION${NC}"

    if [[ "$CUDA_VERSION" == 12.* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == 11.8 ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$CUDA_VERSION" == 11.7 ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    else
        echo -e "${YELLOW}  使用默认 CUDA 版本安装${NC}"
        pip install torch torchvision torchaudio
    fi
else
    echo -e "${YELLOW}  未检测到 CUDA，安装 CPU 版本 PyTorch${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}  PyTorch 安装完成${NC}"
python3 -c "import torch; print(f'  PyTorch 版本: {torch.__version__}'); print(f'  CUDA 可用: {torch.cuda.is_available()}')"

# =============================================================================
# 安装项目依赖
# =============================================================================
echo -e "\n${BLUE}[5/7] 安装项目依赖...${NC}"

pip install --upgrade pip
pip install numpy tqdm datasets tiktoken wandb matplotlib seaborn pytest black flake8

echo -e "${GREEN}  依赖安装完成${NC}"

# =============================================================================
# 安装 Claude Code CLI
# =============================================================================
echo -e "\n${BLUE}[6/7] 安装 Claude Code CLI...${NC}"

if ! command -v claude &> /dev/null; then
    if command -v npm &> /dev/null; then
        npm install -g @anthropic-ai/claude-code
        echo -e "${GREEN}  Claude Code 安装完成${NC}"
    else
        echo -e "${YELLOW}  警告: Node.js 未安装，请手动安装:${NC}"
        echo "    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
        echo "    sudo apt-get install -y nodejs"
        echo "    npm install -g @anthropic-ai/claude-code"
    fi
else
    echo -e "${GREEN}  Claude Code 已安装${NC}"
fi

# =============================================================================
# 创建辅助脚本
# =============================================================================
echo -e "\n${BLUE}[7/7] 创建辅助脚本...${NC}"

# 创建快速启动脚本
cat > "$PROJECT_DIR/start_claude.sh" << 'EOF'
#!/bin/bash
# 启动 Claude Code（自动激活虚拟环境）
cd "$HOME/projects/oelm"
source venv/bin/activate
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
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    显存: {props.total_memory / 1024**3:.1f} GB")

    # 测试 GPU 计算
    if torch.cuda.is_available():
        print("\n测试 GPU 计算...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x.t())
        print(f"  GPU 计算测试: 通过")

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
source venv/bin/activate

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

echo -e "${GREEN}  辅助脚本已创建:${NC}"
echo "    - start_claude.sh: 启动 Claude Code"
echo "    - check_env.py: 检查环境"
echo "    - train_oelm.sh: 快速训练"

# =============================================================================
# 总结
# =============================================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  服务器环境配置完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "项目目录: $PROJECT_DIR"
echo "虚拟环境: $VENV_DIR"
echo ""
echo "常用命令:"
echo "  启动 Claude Code:   ./start_claude.sh"
echo "  检查环境:           python check_env.py"
echo "  训练模型:           ./train_oelm.sh oelm small tinystories"
echo "  运行基准测试:       python benchmark.py --compare"
echo ""
echo "建议下一步:"
echo "  1. 上传你的代码到 $PROJECT_DIR"
echo "  2. 运行 python check_env.py 验证环境"
echo "  3. 运行 ./start_claude.sh 开始开发"
