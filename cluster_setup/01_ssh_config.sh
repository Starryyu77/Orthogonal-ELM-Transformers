#!/bin/bash
# SSH配置和连接指南
# 使用方式: 在本地终端运行以下命令

echo "=== NTU EEE Cluster SSH配置 ==="
echo ""
echo "1. 配置SSH快捷登录 (~/.ssh/config):"
echo ""
cat << 'EOF'
Host ntu-cluster
    HostName 10.97.216.128
    User tianyu016
    Port 22
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF

echo ""
echo "2. 添加上述配置到你的 ~/.ssh/config 文件"
echo ""
echo "3. 连接命令:"
echo "   ssh ntu-cluster"
echo ""
echo "4. 连接后需要:"
echo "   - 输入密码: GoodLuck2025!zty"
echo "   - 首次登录需验证指纹: SHA256:AhYlWQBTsN/4GAzYTVTiZzrVhPhcurFMu1sBMglqvdM"
echo "   - 可能需要修改过期密码"
echo ""
echo "5. 保持会话不中断 (重要):"
echo "   连接后立即运行: tmux new -s oelm"
echo "   如果断开，重新连接后: tmux attach -t oelm"
