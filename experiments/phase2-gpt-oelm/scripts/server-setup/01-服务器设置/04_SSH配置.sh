#!/bin/bash
# =============================================================================
# SSH 连接配置脚本（在本地运行）
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  SSH 连接配置助手${NC}"
echo -e "${BLUE}========================================${NC}"

# =============================================================================
# 检查参数
# =============================================================================
if [ $# -lt 2 ]; then
    echo "用法: $0 <服务器地址> <用户名> [端口]"
    echo ""
    echo "示例:"
    echo "  $0 192.168.1.100 ubuntu"
    echo "  $0 gpu.server.com user 2222"
    echo ""
    exit 1
fi

SERVER=$1
USERNAME=$2
PORT=${3:-22}

SERVER_ALIAS="oelm-server"

echo -e "\n服务器信息:"
echo "  地址: $SERVER"
echo "  用户名: $USERNAME"
echo "  端口: $PORT"

# =============================================================================
# 生成 SSH 密钥（如果不存在）
# =============================================================================
echo -e "\n${BLUE}[1/4] 检查 SSH 密钥...${NC}"

SSH_KEY="$HOME/.ssh/id_ed25519"

if [ ! -f "$SSH_KEY" ]; then
    echo -e "${YELLOW}未检测到 SSH 密钥，正在生成...${NC}"
    ssh-keygen -t ed25519 -C "$(whoami)@$(hostname)" -f "$SSH_KEY" -N ""
    echo -e "${GREEN}SSH 密钥已生成: $SSH_KEY${NC}"
else
    echo -e "${GREEN}SSH 密钥已存在: $SSH_KEY${NC}"
fi

# 显示公钥
echo -e "\n${YELLOW}你的公钥（请复制到服务器的 ~/.ssh/authorized_keys）:${NC}"
echo "----------------------------------------"
cat "${SSH_KEY}.pub"
echo "----------------------------------------"

# =============================================================================
# 复制公钥到服务器（如果可能）
# =============================================================================
echo -e "\n${BLUE}[2/4] 尝试自动复制公钥到服务器...${NC}"

read -p "是否尝试自动复制公钥? (需要输入服务器密码) [y/N]: " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v ssh-copy-id &> /dev/null; then
        ssh-copy-id -p $PORT "${USERNAME}@${SERVER}" || {
            echo -e "${YELLOW}自动复制失败，请手动复制上面的公钥${NC}"
        }
    else
        # 手动复制
        cat "${SSH_KEY}.pub" | ssh -p $PORT "${USERNAME}@${SERVER}" "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys" || {
            echo -e "${YELLOW}自动复制失败，请手动复制上面的公钥${NC}"
        }
    fi
else
    echo -e "${YELLOW}请手动将公钥添加到服务器的 ~/.ssh/authorized_keys${NC}"
fi

# =============================================================================
# 配置 SSH 快捷方式
# =============================================================================
echo -e "\n${BLUE}[3/4] 配置 SSH 快捷方式...${NC}"

SSH_CONFIG="$HOME/.ssh/config"

# 备份原有配置
if [ -f "$SSH_CONFIG" ]; then
    cp "$SSH_CONFIG" "$SSH_CONFIG.backup.$(date +%Y%m%d%H%M%S)"
fi

# 检查是否已配置
if grep -q "Host $SERVER_ALIAS" "$SSH_CONFIG" 2>/dev/null; then
    echo -e "${YELLOW}配置已存在，跳过添加${NC}"
else
    # 添加配置
    mkdir -p "$HOME/.ssh"
    cat >> "$SSH_CONFIG" << EOF

# OELM GPU 服务器
Host $SERVER_ALIAS
    HostName $SERVER
    User $USERNAME
    Port $PORT
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
    # 启用压缩（适合慢速连接）
    Compression yes
EOF
    chmod 600 "$SSH_CONFIG"
    echo -e "${GREEN}SSH 配置已添加到: $SSH_CONFIG${NC}"
fi

# =============================================================================
# 创建辅助脚本
# =============================================================================
echo -e "\n${BLUE}[4/4] 创建快捷脚本...${NC}"

LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"

# 创建快速 SSH 连接脚本
cat > "$LOCAL_BIN/oelm-ssh" << EOF
#!/bin/bash
# 快速连接到 OELM 服务器

# 检查是否需要端口转发（用于 Jupyter/TensorBoard）
if [ "\$1" == "--jupyter" ]; then
    echo "连接到服务器并转发 Jupyter 端口 (8888)..."
    ssh -L 8888:localhost:8888 $SERVER_ALIAS
elif [ "\$1" == "--tensorboard" ]; then
    echo "连接到服务器并转发 TensorBoard 端口 (6006)..."
    ssh -L 6006:localhost:6006 $SERVER_ALIAS
else
    ssh $SERVER_ALIAS
fi
EOF
chmod +x "$LOCAL_BIN/oelm-ssh"

# 创建代码同步脚本
cat > "$LOCAL_BIN/oelm-sync" << 'EOF'
#!/bin/bash
# 同步代码到服务器

SERVER="oelm-server"
LOCAL_DIR="${1:-.}"
REMOTE_DIR="${2:-~/projects/oelm}"

echo "同步 $LOCAL_DIR -> $SERVER:$REMOTE_DIR"
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'venv' \
    --exclude '.claude' \
    --exclude 'out' \
    --exclude '*.pt' \
    --exclude '*.bin' \
    "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"

echo "同步完成"
EOF
chmod +x "$LOCAL_BIN/oelm-sync"

# 创建一键开发脚本
cat > "$LOCAL_BIN/oelm-dev" << 'EOF'
#!/bin/bash
# 一键连接到 OELM 服务器并启动 Claude Code

SERVER="oelm-server"
REMOTE_DIR="~/projects/oelm"

echo "连接到服务器并启动 Claude Code..."
ssh -t $SERVER "cd $REMOTE_DIR && ./start_claude.sh"
EOF
chmod +x "$LOCAL_BIN/oelm-dev"

# 添加到 PATH
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
    echo "export PATH=\"$LOCAL_BIN:\$PATH\"" >> "$HOME/.bashrc"
    echo -e "${YELLOW}已将 $LOCAL_BIN 添加到 PATH${NC}"
    echo -e "${YELLOW}请运行: source ~/.bashrc${NC}"
fi

echo -e "${GREEN}快捷脚本已创建:${NC}"
echo "  oelm-ssh          - 快速 SSH 连接"
echo "  oelm-ssh --jupyter - 带 Jupyter 端口转发"
echo "  oelm-sync         - 同步代码到服务器"
echo "  oelm-dev          - 连接并启动 Claude Code"

# =============================================================================
# 测试连接
# =============================================================================
echo -e "\n${BLUE}测试连接...${NC}"

read -p "是否测试 SSH 连接? [Y/n]: " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "正在连接 $SERVER_ALIAS..."
    ssh -o ConnectTimeout=10 "$SERVER_ALIAS" "echo '连接成功!'; nvidia-smi" || {
        echo -e "${RED}连接失败，请检查:${NC}"
        echo "  1. 服务器地址和端口是否正确"
        echo "  2. 公钥是否正确添加到服务器"
        echo "  3. 服务器 SSH 服务是否运行"
    }
fi

# =============================================================================
# 总结
# =============================================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  SSH 配置完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "快捷命令:"
echo "  oelm-ssh          - SSH 连接到服务器"
echo "  oelm-sync [本地] [远程] - 同步代码"
echo "  oelm-dev          - 连接并启动 Claude Code"
echo ""
echo "手动连接:"
echo "  ssh $SERVER_ALIAS"
echo ""
echo "VS Code Remote:"
echo "  1. 安装 Remote-SSH 扩展"
echo "  2. 选择 'Connect to Host...'"
echo "  3. 选择 '$SERVER_ALIAS'"
