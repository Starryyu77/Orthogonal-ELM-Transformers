#!/bin/bash
# =============================================================================
# NTU GPU 服务器配置脚本
# FQDN: gpu43.dynip.ntu.edu.sg
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NTU GPU 服务器配置${NC}"
echo -e "${BLUE}========================================${NC}"

# 服务器信息
SERVER="gpu43.dynip.ntu.edu.sg"
USER="s125mdg43_10"
ALIAS="ntu-gpu43"

# =============================================================================
# 检查 SSH 密钥
# =============================================================================
echo -e "\n${BLUE}[1/3] 检查 SSH 密钥...${NC}"

SSH_DIR="$HOME/.ssh"
SSH_KEY="$SSH_DIR/id_ed25519"

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

if [ ! -f "$SSH_KEY" ]; then
    echo -e "${YELLOW}生成新的 SSH 密钥...${NC}"
    ssh-keygen -t ed25519 -C "$USER@ntu" -f "$SSH_KEY" -N ""
    echo -e "${GREEN}SSH 密钥已生成${NC}"
else
    echo -e "${GREEN}SSH 密钥已存在${NC}"
fi

echo -e "\n${YELLOW}你的公钥（需要在服务器上注册）:${NC}"
echo "----------------------------------------"
cat "${SSH_KEY}.pub"
echo "----------------------------------------"

# =============================================================================
# 配置 SSH 快捷方式
# =============================================================================
echo -e "\n${BLUE}[2/3] 配置 SSH 快捷方式...${NC}"

SSH_CONFIG="$SSH_DIR/config"

# 备份原有配置
if [ -f "$SSH_CONFIG" ]; then
    cp "$SSH_CONFIG" "$SSH_CONFIG.backup.$(date +%Y%m%d%H%M%S)"
fi

# 检查是否已配置
if grep -q "Host $ALIAS" "$SSH_CONFIG" 2>/dev/null; then
    echo -e "${YELLOW}配置已存在，正在更新...${NC}"
    # 删除旧配置
    sed -i.bak "/Host $ALIAS/,/^Host/d" "$SSH_CONFIG" 2>/dev/null || true
fi

# 添加配置
cat >> "$SSH_CONFIG" << EOF

# NTU GPU 服务器
Host $ALIAS
    HostName $SERVER
    User $USER
    Port 22
    ServerAliveInterval 60
    ServerAliveCountMax 3
    Compression yes
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
EOF

chmod 600 "$SSH_CONFIG"
echo -e "${GREEN}SSH 配置已添加${NC}"

# =============================================================================
# 创建快捷命令
# =============================================================================
echo -e "\n${BLUE}[3/3] 创建快捷命令...${NC}"

LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"

# 快速 SSH
cat > "$LOCAL_BIN/ntu-ssh" << 'EOF'
#!/bin/bash
# 连接到 NTU GPU 服务器

SERVER="ntu-gpu43"

if [ "$1" == "jupyter" ]; then
    echo "连接并转发 Jupyter 端口..."
    ssh -L 8888:localhost:8888 $SERVER
elif [ "$1" == "tensorboard" ]; then
    echo "连接并转发 TensorBoard 端口..."
    ssh -L 6006:localhost:6006 $SERVER
else
    ssh $SERVER
fi
EOF
chmod +x "$LOCAL_BIN/ntu-ssh"

# 同步代码
cat > "$LOCAL_BIN/ntu-sync" << 'EOF'
#!/bin/bash
# 同步代码到 NTU GPU 服务器

SERVER="ntu-gpu43"
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
    --exclude '.DS_Store' \
    --exclude 'data/*.bin' \
    "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"

echo "同步完成"
EOF
chmod +x "$LOCAL_BIN/ntu-sync"

# 一键启动 Claude Code
cat > "$LOCAL_BIN/ntu-claude" << 'EOF'
#!/bin/bash
# 连接到服务器并启动 Claude Code

SERVER="ntu-gpu43"
REMOTE_DIR="~/projects/oelm"

echo "连接到 NTU GPU 服务器并启动 Claude Code..."
ssh -t $SERVER "cd $REMOTE_DIR 2>/dev/null || mkdir -p $REMOTE_DIR; cd $REMOTE_DIR; if [ -f start_claude.sh ]; then ./start_claude.sh; else echo '请先在服务器上运行环境配置脚本'; bash; fi"
EOF
chmod +x "$LOCAL_BIN/ntu-claude"

# 服务器环境安装脚本
cat > "$LOCAL_BIN/ntu-setup" << 'EOF'
#!/bin/bash
# 在 NTU 服务器上安装环境

SERVER="ntu-gpu43"
SETUP_SCRIPT="$(dirname "$0")/../setup_server.sh"

echo "上传环境配置脚本到服务器..."
scp "$SETUP_SCRIPT" $SERVER:~/setup_server.sh

echo "在服务器上运行配置..."
ssh -t $SERVER "bash ~/setup_server.sh"
EOF
chmod +x "$LOCAL_BIN/ntu-setup"

# 添加到 PATH
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
    echo "" >> "$HOME/.bashrc"
    echo "# NTU GPU 工具" >> "$HOME/.bashrc"
    echo "export PATH=\"$LOCAL_BIN:\$PATH\"" >> "$HOME/.zshrc" 2>/dev/null || true
    echo "export PATH=\"$LOCAL_BIN:\$PATH\"" >> "$HOME/.bashrc"
    echo -e "${YELLOW}已将工具添加到 PATH${NC}"
fi

# =============================================================================
# 总结
# =============================================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  配置完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "服务器信息:"
echo "  地址: $SERVER"
echo "  用户名: $USER"
echo "  别名: $ALIAS"
echo ""
echo "快捷命令:"
echo "  ntu-ssh           - SSH 连接到服务器"
echo "  ntu-ssh jupyter   - 带 Jupyter 端口转发"
echo "  ntu-sync          - 同步代码到服务器"
echo "  ntu-claude        - 连接并启动 Claude Code"
echo "  ntu-setup         - 在服务器上安装环境"
echo ""
echo "首次连接步骤:"
echo "  1. ntu-ssh        (输入密码: ADeaTHStARB)"
echo "  2. 在服务器上粘贴你的公钥到 ~/.ssh/authorized_keys"
echo "  3. ntu-setup      (安装 Python 环境)"
echo "  4. ntu-claude     (开始开发)"
echo ""
echo "手动连接:"
echo "  ssh $ALIAS"
