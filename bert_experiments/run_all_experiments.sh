#!/bin/bash

# BERT-ELM实验批量运行脚本
# 用法: bash run_all_experiments.sh

echo "========================================"
echo "BERT-ELM实验批量运行脚本"
echo "========================================"
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python。请先安装Python 3.8+"
    exit 1
fi

echo "Python版本:"
python --version
echo ""

# 检查是否有GPU
echo "检查GPU可用性..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}') if torch.cuda.is_available() else print('未检测到GPU，将使用CPU')"
echo ""

# 创建日志目录
mkdir -p logs
echo "日志将保存到 logs/ 目录"
echo ""

# 运行IMDB实验
echo "========================================"
echo "1/4 启动IMDB实验..."
echo "========================================"
nohup python bert_imdb_experiments.py > logs/imdb_log.txt 2>&1 &
IMDB_PID=$!
echo "IMDB实验已启动 (PID: $IMDB_PID)"
echo "日志文件: logs/imdb_log.txt"
echo ""

# 等待一段时间，避免同时加载模型导致OOM
sleep 5

# 运行AG News实验
echo "========================================"
echo "2/4 启动AG News实验..."
echo "========================================"
nohup python bert_agnews_experiments.py > logs/agnews_log.txt 2>&1 &
AGNEWS_PID=$!
echo "AG News实验已启动 (PID: $AGNEWS_PID)"
echo "日志文件: logs/agnews_log.txt"
echo ""

sleep 5

# 运行MNLI实验
echo "========================================"
echo "3/4 启动MNLI实验..."
echo "========================================"
nohup python bert_mnli_experiments.py > logs/mnli_log.txt 2>&1 &
MNLI_PID=$!
echo "MNLI实验已启动 (PID: $MNLI_PID)"
echo "日志文件: logs/mnli_log.txt"
echo ""

sleep 5

# 运行XNLI实验
echo "========================================"
echo "4/4 启动XNLI实验（英语）..."
echo "========================================"
nohup python bert_xnli_experiments.py --language en > logs/xnli_log.txt 2>&1 &
XNLI_PID=$!
echo "XNLI实验已启动 (PID: $XNLI_PID)"
echo "日志文件: logs/xnli_log.txt"
echo ""

# 保存PID到文件
echo $IMDB_PID > logs/imdb.pid
echo $AGNEWS_PID > logs/agnews.pid
echo $MNLI_PID > logs/mnli.pid
echo $XNLI_PID > logs/xnli.pid

echo "========================================"
echo "所有实验已成功启动！"
echo "========================================"
echo ""
echo "进程ID已保存到 logs/*.pid"
echo ""
echo "监控实验进度:"
echo "  tail -f logs/imdb_log.txt"
echo "  tail -f logs/agnews_log.txt"
echo "  tail -f logs/mnli_log.txt"
echo "  tail -f logs/xnli_log.txt"
echo ""
echo "检查进程状态:"
echo "  ps aux | grep bert_.*_experiments.py | grep -v grep"
echo ""
echo "停止所有实验:"
echo "  kill $IMDB_PID $AGNEWS_PID $MNLI_PID $XNLI_PID"
echo ""
echo "预计完成时间: 8-12小时（取决于GPU性能）"
