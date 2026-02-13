#!/bin/bash
# 环境配置脚本 - 使用项目目录作为conda环境位置
# 在服务器上运行

echo "=== OELM 环境配置 (项目目录版) ==="
echo ""

# 进入项目目录
cd /projects/oelmexperiment

# 设置conda环境路径为项目目录
export CONDA_ENVS_PATH=/projects/oelmexperiment/.conda/envs
export CONDA_PKGS_DIRS=/projects/oelmexperiment/.conda/pkgs

# 创建conda目录
mkdir -p /projects/oelmexperiment/.conda/{envs,pkgs}

echo "1. Conda环境将创建在: /projects/oelmexperiment/.conda/envs"
echo ""

# 加载Miniconda
module load Miniconda3
source activate

# 设置conda使用项目目录
echo "2. 配置conda使用项目目录..."
conda config --add envs_dirs /projects/oelmexperiment/.conda/envs
conda config --add pkgs_dirs /projects/oelmexperiment/.conda/pkgs

echo ""
echo "3. 创建oelm环境 (这可能需要几分钟)..."
conda create -p /projects/oelmexperiment/.conda/envs/oelm python=3.10 -y

echo ""
echo "4. 激活环境..."
source activate /projects/oelmexperiment/.conda/envs/oelm

echo ""
echo "5. 安装PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "6. 安装其他依赖..."
pip install transformers datasets tiktoken wandb tqdm numpy scipy

echo ""
echo "7. 验证安装..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"

echo ""
echo "=== 环境配置完成 ==="
echo ""
echo "以后激活环境使用:"
echo "  source activate /projects/oelmexperiment/.conda/envs/oelm"
echo ""
echo "或者添加到 ~/.bashrc:"
echo "  alias oelm='source activate /projects/oelmexperiment/.conda/envs/oelm'"
