#!/bin/bash
# 快速开始指南 - 在服务器上按顺序执行这些命令

echo "=== OELM集群实验快速开始 ==="
echo ""

# Step 1: 创建项目目录
echo "Step 1: 创建项目目录..."
mkdir -p /projects/oelm-experiment
cd /projects/oelm-experiment

# Step 2: 使用storagemgr分配存储 (交互式)
echo ""
echo "Step 2: 运行 storagemgr 分配存储 (需要交互操作)..."
echo "   运行: storagemgr"
echo "   选择: Create new project directory"
echo "   名称: oelm-experiment"
echo "   配额: 500GB"

# Step 3: 上传代码 (在本地执行)
echo ""
echo "Step 3: 在本地终端上传代码..."
echo "   cd /Users/starryyu/Library/Mobile Documents/com~apple~CloudDocs/学术研究/Orthogonal ELM Transformers/Train"
echo "   scp -r . ntu-cluster:/projects/oelm-experiment/"

# Step 4: 配置环境
echo ""
echo "Step 4: 配置环境..."
echo "   bash cluster_setup/03_setup_env.sh"

# Step 5: 准备数据
echo ""
echo "Step 5: 准备数据 (提交后台任务)..."
echo "   sbatch cluster_setup/04_prepare_data.sh"

# Step 6: 训练GPT基线
echo ""
echo "Step 6: 训练GPT基线..."
echo "   sbatch cluster_setup/05_train_gpt.slurm"

# Step 7: 训练OELM
echo ""
echo "Step 7: 训练OELM..."
echo "   sbatch cluster_setup/06_train_oelm.slurm"

# Step 8: 查看状态
echo ""
echo "Step 8: 查看任务状态..."
echo "   squeue -u tianyu016"

echo ""
echo "=== 完成 ==="
