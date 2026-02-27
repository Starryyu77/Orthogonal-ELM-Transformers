#!/bin/bash

# 快速测试脚本 - 验证环境和代码是否正常工作
# 使用小样本快速测试所有脚本

echo "========================================"
echo "BERT-ELM环境和脚本快速测试"
echo "========================================"
echo ""

# 测试配置
EPOCHS=1
NUM_EXP=1
TRAIN_SAMPLES=100
TEST_SAMPLES=50
BATCH_SIZE=8

echo "测试配置:"
echo "  训练样本: $TRAIN_SAMPLES"
echo "  测试样本: $TEST_SAMPLES"
echo "  批次大小: $BATCH_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  重复次数: $NUM_EXP"
echo ""

# 测试IMDB
echo "========================================"
echo "测试 1/4: IMDB"
echo "========================================"
python bert_imdb_experiments.py \
  --epochs $EPOCHS \
  --num-exp $NUM_EXP \
  --train-samples $TRAIN_SAMPLES \
  --test-samples $TEST_SAMPLES \
  --batch-size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo "✓ IMDB测试通过"
else
    echo "✗ IMDB测试失败"
    exit 1
fi
echo ""

# 测试AG News
echo "========================================"
echo "测试 2/4: AG News"
echo "========================================"
python bert_agnews_experiments.py \
  --epochs $EPOCHS \
  --num-exp $NUM_EXP \
  --train-samples $TRAIN_SAMPLES \
  --test-samples $TEST_SAMPLES \
  --batch-size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo "✓ AG News测试通过"
else
    echo "✗ AG News测试失败"
    exit 1
fi
echo ""

# 测试MNLI
echo "========================================"
echo "测试 3/4: MNLI"
echo "========================================"
python bert_mnli_experiments.py \
  --epochs $EPOCHS \
  --num-exp $NUM_EXP \
  --train-samples $TRAIN_SAMPLES \
  --test-samples $TEST_SAMPLES \
  --batch-size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo "✓ MNLI测试通过"
else
    echo "✗ MNLI测试失败"
    exit 1
fi
echo ""

# 测试XNLI
echo "========================================"
echo "测试 4/4: XNLI"
echo "========================================"
python bert_xnli_experiments.py \
  --language en \
  --epochs $EPOCHS \
  --num-exp $NUM_EXP \
  --train-samples $TRAIN_SAMPLES \
  --test-samples $TEST_SAMPLES \
  --batch-size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo "✓ XNLI测试通过"
else
    echo "✗ XNLI测试失败"
    exit 1
fi
echo ""

echo "========================================"
echo "所有测试通过! ✓"
echo "========================================"
echo ""
echo "环境配置正确，可以运行完整实验。"
echo "使用以下命令运行完整实验:"
echo "  bash run_all_experiments.sh"
