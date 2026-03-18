# OELM Pre-training Experiments

**Goal**: 验证OELM（正交Q/K冻结）在预训练阶段的效果

## 实验设计

### 核心假设

| 场景 | 微调 | 预训练 |
|------|------|--------|
| Q/K来源 | 已训练好的表示 | 随机/正交初始化 |
| 冻结效果 | 保持已有知识 | 提供稳定初始化？ |
| 潜在价值 | 参数效率 | 训练稳定性+正则化 |

### 实验配置

| 方法 | 冻结Q/K | 冻结FFN | 可训练参数 | 学习率 |
|------|---------|---------|-----------|--------|
| Baseline | ❌ | ❌ | 100% | 3e-4 |
| OELM-QK | ✅ | ❌ | ~75% | 1e-3 |
| OELM-QK-FFN | ✅ | ✅ | ~65% | 1e-3 |

### 模型规模

| 配置 | d_model | layers | heads | 参数量 |
|------|---------|--------|-------|--------|
| Small | 768 | 12 | 12 | 117M |
| Medium | 1024 | 24 | 16 | 355M |

## 快速开始

### 1. 设置环境（首次运行）

```bash
# SSH到集群
ssh ntu-cluster

# 进入项目目录
cd /projects/LlamaFactory/OELM-Pretrain

# 设置环境
./scripts/setup_env.sh
```

### 2. 提交预训练任务

```bash
# 提交所有预训练实验（并行运行）
./scripts/submit_all.sh

# 或单独提交
sbatch scripts/run_pretrain_baseline.sh
sbatch scripts/run_pretrain_oelm_qk.sh
sbatch scripts/run_pretrain_oelm_qk_ffn.sh
```

### 3. 监控训练

```bash
# 查看作业状态
watch squeue -u tianyu016

# 查看日志
tail -f logs/pretrain_*.out
```

### 4. 下游微调（预训练完成后）

```bash
# 微调预训练模型
sbatch scripts/run_finetune.sh \
    outputs/pretrain/oelm_qk/best/pytorch_model.pt \
    imdb \
    oelm_qk
```

## 目录结构

```
OELM-Pretrain/
├── README.md
├── models/
│   └── modeling_oelm_pretrain.py    # 模型定义
├── scripts/
│   ├── train_pretrain.py            # 预训练脚本
│   ├── finetune_from_pretrain.py    # 微调脚本
│   ├── run_pretrain_baseline.sh     # Baseline启动脚本
│   ├── run_pretrain_oelm_qk.sh      # OELM-QK启动脚本
│   ├── run_pretrain_oelm_qk_ffn.sh  # OELM-QK-FFN启动脚本
│   ├── run_finetune.sh              # 微调启动脚本
│   ├── submit_all.sh                # 一键提交所有预训练
│   └── setup_env.sh                 # 环境设置
├── data/                            # 数据缓存
├── outputs/                         # 输出结果
│   ├── pretrain/                    # 预训练模型
│   └── finetune/                    # 微调结果
├── checkpoints/                     # 检查点
└── logs/                            # 训练日志
```

## 预期结果

### 成功标准

1. **预训练PPL**: OELM方法与baseline差距 < 10%
2. **下游任务**: 微调后准确率相当或更好
3. **参数效率**: 用更少参数达到相似性能

### 预期时间

| 实验 | 数据集 | 预计时间 |
|------|--------|----------|
| Pretrain Small | TinyStories (1 epoch) | 4-8小时 |
| Finetune | IMDB | 30分钟 |

## 评估指标

### 预训练

- **Loss**: Cross-entropy loss
- **PPL (Perplexity)**: exp(loss)
- **训练速度**: tokens/second

### 下游微调

- **Accuracy**: 分类准确率
- **F1 Score**: 加权F1

## 与之前实验的关系

| Phase | 任务 | 结果 |
|-------|------|------|
| Phase 1-4 | 微调分类 | ✅ OELM有效 |
| Phase 2-3 | 预训练生成 | ❌ OELM无效 |
| **Phase 5** | **预训练→微调** | ❓ 验证中 |

**核心问题**: 预训练阶段的Q/K冻结是否比微调阶段更有价值？

## 相关文档

- 项目总览: `Train/README.md`
- Phase 4报告: `Train/experiments/phase4-gpt-classification/REPORT.md`
- OELM-FFN结果: `EXPERIMENT_RESULTS_OELM_FFN.md`

## 故障排除

### 常见问题

1. **CUDA out of memory**
   - 减小batch_size
   - 减小max_seq_len

2. **Dataset download slow**
   - 使用cache_dir缓存数据集
   - 提前下载数据

3. **Job排队时间长**
   - 尝试其他GPU类型
   - 使用`--qos override-limits-but-killable`

## 联系

- 项目负责人: tianyu016
- 更新时间: 2026-03-09