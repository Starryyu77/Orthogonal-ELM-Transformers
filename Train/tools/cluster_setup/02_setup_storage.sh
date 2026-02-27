#!/bin/bash
# 存储配置脚本
# 在服务器上运行此脚本

echo "=== OELM 项目存储配置 ==="
echo ""

# 1. 创建项目目录
echo "1. 创建项目目录..."
mkdir -p /projects/oelm-experiment
cd /projects/oelm-experiment

# 2. 使用storagemgr扩展存储
echo ""
echo "2. 运行 storagemgr 扩展存储配额"
echo "   请按以下步骤操作:"
echo ""
echo "   a) 运行: storagemgr"
echo "   b) 选择 'Create new project directory'"
echo "   c) 输入名称: oelm-experiment"
echo "   d) 分配空间: 500GB (或根据需求调整)"
echo ""

# 3. 创建子目录结构
echo "3. 创建实验目录结构..."
mkdir -p data/{tiny_stories,wikitext103,openwebtext}
mkdir -p models/checkpoints
mkdir -p logs/{gpt,oelm}
mkdir -p results
mkdir -p scripts/slurm

echo ""
echo "目录结构已创建:"
tree -L 2 /projects/oelm-experiment 2>/dev/null || ls -R /projects/oelm-experiment

echo ""
echo "=== 存储配置完成 ==="
echo "项目路径: /projects/oelm-experiment"
