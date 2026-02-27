# NTU MLDA GPU 服务器 - 新项目快速配置指南

## 一键设置脚本

在新项目根目录创建 `setup_ntu.sh`：

```bash
#!/bin/bash
# =============================================================================
# NTU MLDA GPU 服务器项目设置脚本
# 放置在新项目根目录并运行: bash setup_ntu.sh
# =============================================================================

set -e

echo "========================================"
echo "  NTU MLDA GPU 项目设置"
echo "========================================"

# 项目名称
PROJECT_NAME=$(basename "$PWD")
echo "项目名称: $PROJECT_NAME"

# 复制 ntu-run.sh 脚本
cat > ntu-run.sh << 'SCRIPT_EOF'
#!/bin/bash
# =============================================================================
# NTU MLDA GPU 服务器远程运行脚本
# =============================================================================

SERVER="ntu-gpu43"
PROJECT_NAME="PROJECT_NAME_PLACEHOLDER"
PROJECT_DIR="~/projects/$PROJECT_NAME"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 构建远程命令
REMOTE_CMD="cd $PROJECT_DIR && source venv/bin/activate 2>/dev/null || true"

show_help() {
    echo "NTU MLDA GPU 服务器远程运行工具"
    echo "项目: $PROJECT_NAME"
    echo ""
    echo "用法: $0 <命令> [参数]"
    echo ""
    echo "可用命令:"
    echo "  setup               在服务器上设置项目环境"
    echo "  train [args]        运行训练 (使用 GPU 2)"
    echo "  train-multi [n]     多 GPU 训练 (使用 GPU 2,3)"
    echo "  python <args>       运行 Python 命令"
    echo "  bash                交互式 Bash"
    echo "  sync                同步代码到服务器"
    echo "  status              查看 GPU 状态"
    echo "  jupyter             启动 Jupyter Lab"
    echo "  tensorboard         启动 TensorBoard"
    echo ""
    echo "示例:"
    echo "  $0 setup"
    echo "  $0 train --epochs 10"
    echo "  $0 train-multi 2"
    echo "  $0 python test.py"
}

case "$1" in
    setup)
        echo -e "${BLUE}在服务器上设置项目...${NC}"
        ssh $SERVER "mkdir -p $PROJECT_DIR && cd $PROJECT_DIR && python3 -m venv venv && source venv/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install numpy tqdm matplotlib"
        echo -e "${GREEN}设置完成!${NC}"
        ;;

    train)
        shift
        echo -e "${BLUE}启动训练 (GPU 2)...${NC}"
        ssh $SERVER "$REMOTE_CMD && export CUDA_VISIBLE_DEVICES=2 && python train.py $@"
        ;;

    train-multi)
        NUM_GPUS=${2:-2}
        shift 2
        echo -e "${BLUE}启动多 GPU 训练 ($NUM_GPUS GPUs)...${NC}"
        ssh $SERVER "$REMOTE_CMD && export CUDA_VISIBLE_DEVICES=2,3 && torchrun --standalone --nproc_per_node=$NUM_GPUS train.py $@"
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

    sync)
        echo -e "${BLUE}同步代码到服务器...${NC}"
        rsync -avz --progress \
            --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
            --exclude 'venv' --exclude '.claude' --exclude 'out' \
            --exclude '*.pt' --exclude '*.bin' --exclude '.DS_Store' \
            --exclude 'data/*.bin' \
            "." "$SERVER:$PROJECT_DIR/"
        echo -e "${GREEN}同步完成${NC}"
        ;;

    status)
        echo -e "${BLUE}GPU 状态:${NC}"
        ssh $SERVER "nvidia-smi"
        ;;

    jupyter)
        echo -e "${BLUE}启动 Jupyter Lab (http://localhost:8888)...${NC}"
        ssh -L 8888:localhost:8888 -t $SERVER "$REMOTE_CMD && jupyter lab --ip=0.0.0.0 --no-browser --port=8888"
        ;;

    tensorboard)
        echo -e "${BLUE}启动 TensorBoard (http://localhost:6006)...${NC}"
        ssh -L 6006:localhost:6006 -t $SERVER "$REMOTE_CMD && tensorboard --logdir=out --port=6006"
        ;;

    *)
        show_help
        ;;
esac
SCRIPT_EOF

# 替换项目名
sed -i.bak "s/PROJECT_NAME_PLACEHOLDER/$PROJECT_NAME/g" ntu-run.sh
rm -f ntu-run.sh.bak

chmod +x ntu-run.sh

echo ""
echo "✅ ntu-run.sh 已创建"
echo ""
echo "下一步:"
echo "  1. ./ntu-run.sh setup    # 在服务器上设置环境"
echo "  2. ./ntu-run.sh sync     # 同步代码"
echo "  3. ./ntu-run.sh train    # 开始训练"
```

## 使用方法

### 新项目快速开始

```bash
# 1. 进入你的新项目
cd /path/to/your-new-project

# 2. 复制上面的 setup_ntu.sh 内容，保存并运行
bash setup_ntu.sh

# 3. 在服务器上设置环境
./ntu-run.sh setup

# 4. 同步代码并开始训练
./ntu-run.sh sync
./ntu-run.sh train --epochs 10
```

### 通用 ntu-run.sh 命令

创建后，所有项目使用相同的命令模式：

```bash
./ntu-run.sh setup          # 服务器环境初始化
./ntu-run.sh sync           # 同步代码
./ntu-run.sh train          # 单卡训练
./ntu-run.sh train-multi 2  # 双卡训练
./ntu-run.sh status         # 查看 GPU
./ntu-run.sh bash           # 交互式终端
```

## 从 OELM 项目复制

如果设置脚本不可用，直接复制：

```bash
# 从 OELM 项目复制
SOURCE="/Users/starryyu/Library/Mobile Documents/com~apple~CloudDocs/学术研究/Orthogonal ELM Transformers/参考材料/Train"
TARGET="/path/to/new-project"

cp "$SOURCE/ntu-run.sh" "$TARGET/"
cd "$TARGET"

# 修改项目名
PROJECT_NAME=$(basename "$PWD")
sed -i.bak "s/oelm_multi/$PROJECT_NAME/g" ntu-run.sh
rm -f ntu-run.sh.bak

chmod +x ntu-run.sh
```

## 服务器账户信息（备忘）

```
服务器: gpu43.dynip.ntu.edu.sg
用户名: s125mdg43_10
密码:   ADeaTHStARB
GPU:    4x RTX A5000 (使用 GPU 2,3)
```
