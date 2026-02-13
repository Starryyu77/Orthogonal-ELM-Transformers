# Phase 1: BERT XNLI 实验

> 验证 OELM (冻结 Q/K + 正交初始化) 在 BERT 编码器上的有效性

---

## 实验目标

1. 验证分头正交初始化在BERT上的有效性
2. 对比 Baseline vs OELM-Freeze 的性能
3. 测量训练速度和参数效率
4. 为后续GPT移植提供验证基础

---

## 实验设计

### 数据集

| 属性 | 详情 |
|------|------|
| 名称 | XNLI (Cross-lingual NLI) |
| 任务 | 自然语言推理 (3分类) |
| 语言 | 英语 (en) |
| 训练集 | 392,702 (使用MNLI) |
| 验证集 | 2,490 |
| 类别 | 蕴含 / 中立 / 矛盾 |

### 模型配置

| 配置 | 值 |
|------|-----|
| 模型 | bert-base-uncased |
| 隐藏层维度 | 768 |
| 注意力头数 | 12 |
| 层数 | 12 |
| 总参数量 | 109.5M |

### 训练配置

| 配置 | Baseline | OELM-Freeze |
|------|----------|-------------|
| 学习率 | 2e-5 | 1e-4 |
| Batch Size | 32 → 16* | 32 → 16* |
| Epochs | 3 | 3 |
| 优化器 | AdamW | AdamW |
| 冻结Q/K | ❌ | ✅ |
| 初始化 | 标准随机 | 分头正交 |

*注：因OOM问题从32调整为16

---

## 实验结果

### 性能指标

| 指标 | Baseline | OELM-Freeze | 对比 |
|------|----------|-------------|------|
| **最佳准确率** | 76.71% | **77.79%** | **+1.08%** ✅ |
| 最终Loss | 0.523 | 0.498 | -4.8% |
| 可训练参数 | 109.5M (100%) | 95.3M (87.1%) | 节省 12.9% |

### 训练效率

| 指标 | Baseline | OELM-Freeze | 对比 |
|------|----------|-------------|------|
| **纯训练时间** | 11,971s | **5,138s** | **-57.1%** ⭐ |
| **每步时间** | 0.162s | **0.069s** | **-57.2%** ⭐ |
| 总训练时间 | 3h 48m | 4h 02m | +6.2% |

### 关键发现

✅ **OELM在BERT上非常有效**
- 准确率提升 +1.08%
- 训练速度提升 57%
- 参数节省 12.9%

---

## 目录结构

```
phase1-bert-xnli/
├── README.md              # 本文件
├── REPORT.md              # 详细实验报告
├── models/
│   ├── modeling_bert_oelm.py   # OELM BERT模型
│   └── train_bert.py           # 训练脚本
├── scripts/
│   ├── run_xnli_experiments.sh    # XNLI实验启动
│   ├── run_fair_comparison.sh     # 公平对比
│   └── run_xnli_oelm_restart.sh   # 重启脚本
├── docs/
│   ├── EXPERIMENT_REPORT_BERT_RESERVOIR.md
│   └── EXPERIMENT_REPORT_BERT_RESERVOIR.pdf
└── logs/
    ├── COMPARISON_EXPERIMENT_LOG.md
    └── TRAINING_LOG.md
```

---

## 使用方法

### 启动实验

```bash
# 在远程服务器上执行
cd ~/Orthogonal_ELM_Transformers/Train/experiments/phase1-bert-xnli

# 启动完整实验
./scripts/run_xnli_experiments.sh

# 公平对比实验
./scripts/run_fair_comparison.sh
```

### 单独运行

```bash
# Baseline
python models/train_bert.py \
    --dataset xnli \
    --freeze_mode false \
    --lr 2e-5 \
    --batch_size 16 \
    --epochs 3

# OELM-Freeze
python models/train_bert.py \
    --dataset xnli \
    --freeze_mode true \
    --lr 1e-4 \
    --batch_size 16 \
    --epochs 3 \
    --init_method orthogonal
```

---

## 下一步

→ 详见 [`../phase2-gpt-oelm/`](../phase2-gpt-oelm/) - GPT移植实验

---

**实验完成时间**: 2026-02-08
