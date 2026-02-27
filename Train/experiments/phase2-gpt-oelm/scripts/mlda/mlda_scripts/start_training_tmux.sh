#!/bin/bash
# 使用 tmux 启动训练（断开SSH后训练继续）

echo "=== 使用 tmux 启动训练 ==="
echo ""
echo "说明:"
echo "  - 训练将在后台运行，断开SSH不受影响"
echo "  - 重新连接后使用: tmux attach -t oelm_gpt 查看GPT训练"
echo "  - 重新连接后使用: tmux attach -t oelm_oelm 查看OELM训练"
echo ""

# 创建GPT训练会话
tmux new-session -d -s oelm_gpt
sleep 2
tmux send-keys -t oelm_gpt "cd ~/Orthogonal_ELM_Transformers/Train && bash mlda_scripts/train_gpt_medium512.sh" Enter
echo "✓ GPT训练已在 tmux 会话 'oelm_gpt' 中启动"

# 创建OELM训练会话
tmux new-session -d -s oelm_oelm
sleep 2
tmux send-keys -t oelm_oelm "cd ~/Orthogonal_ELM_Transformers/Train && bash mlda_scripts/train_oelm_medium512.sh" Enter
echo "✓ OELM训练已在 tmux 会话 'oelm_oelm' 中启动"

echo ""
echo "查看训练状态:"
echo "  tmux ls                    # 列出所有会话"
echo "  tmux attach -t oelm_gpt    # 查看GPT训练"
echo "  tmux attach -t oelm_oelm   # 查看OELM训练"
echo ""
echo "在tmux内操作:"
echo "  Ctrl+b d  # 分离会话（后台继续运行）"
echo "  Ctrl+b [  # 进入滚动模式（上下键查看历史）"
echo "  q         # 退出滚动模式"
