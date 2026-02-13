#!/bin/bash
# 环境配置脚本
# 在服务器上运行

echo "=== OELM 环境配置 ==="
echo ""

# 1. 加载Miniconda
echo "1. 加载Miniconda..."
module load Miniconda3
source activate

# 2. 创建conda环境
echo ""
echo "2. 创建oelm环境..."
conda create -n oelm python=3.10 -y

# 3. 激活环境
echo ""
echo "3. 激活环境..."
source activate oelm

# 4. 安装依赖
echo ""
echo "4. 安装PyTorch和其他依赖..."

# PyTorch (根据集群GPU选择CUDA版本)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 其他依赖
pip install transformers datasets tiktoken wandb tqdm numpy scipy
pip install pytest black flake8

echo ""
echo "5. 验证安装..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"

echo ""
echo "=== 环境配置完成 ==="
echo "激活环境命令: source activate oelm"
