#!/bin/bash
# 数据准备脚本
#SBATCH --job-name=data_prep
#SBATCH --time=4:00:00
#SBATCH --output=logs/data_prep_%j.out
#SBATCH --error=logs/data_prep_%j.err
#SBATCH --partition=cluster02

set -e

echo "=== 数据准备任务 ==="
echo "开始时间: $(date)"

# 激活venv环境
source /projects/oelmexperiment/.venv/bin/activate

# 进入项目目录
cd /projects/oelmexperiment

# 1. TinyStories (快速测试用)
echo "1. 准备 TinyStories 数据集..."
python data/prepare_data.py \
    --dataset tiny_stories \
    --output_dir data/tiny_stories \
    --tokenizer gpt2

# 2. WikiText-103 (主要评估数据集)
echo "2. 准备 WikiText-103 数据集..."
python data/prepare_data.py \
    --dataset wikitext103 \
    --output_dir data/wikitext103 \
    --tokenizer gpt2

# 3. OpenWebText 子集 (通用能力验证)
echo "3. 准备 OpenWebText 子集 (100k文档)..."
python data/prepare_data.py \
    --dataset openwebtext \
    --output_dir data/openwebtext_100k \
    --tokenizer gpt2 \
    --max_docs 100000

echo ""
echo "数据准备完成: $(date)"
echo ""
echo "数据位置:"
ls -lh data/*/
