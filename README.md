# Orthogonal ELM Transformers (OELM)

> **Efficient Training of Transformers via Orthogonal Initialization and Parameter Freezing**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🔬 核心原理

OELM (Orthogonal Extreme Learning Machine) 是一种高效的 Transformer 训练方法，通过以下技术减少可训练参数并提升性能：

### 1. 分头正交初始化 (Head-wise Orthogonal Initialization)
- 对每个注意力头的 Q/K 投影矩阵独立进行 QR 分解
- 保持注意力头的多样性和几何结构
- 避免全局正交初始化破坏头间独立性

### 2. 参数冻结策略
- **OELM-QK**: 冻结 Query/Key 投影，仅训练 Value/Output
  - 可训练参数：~75%
  - 保留正交初始化后的稳定注意力模式
  
- **OELM-QK-FFN**: 额外冻结 FFN 前馈网络
  - 可训练参数：~65%
  - 更强的正则化效果

### 3. 理论优势
- **减少过拟合**: 冻结参数提供隐式正则化
- **训练加速**: 更少的梯度计算，更快的收敛
- **性能提升**: 在分类任务上平均提升 +10.46%

---

## 📊 实验总览

| 实验 | 名称 | 数据集 | 核心发现 | 完整报告 |
|:-----|:-----|:-------|:---------|:---------|
| **Exp 1** | BERT XNLI | XNLI (3分类) | OELM 优于 Baseline (+1.08%)，训练快 57% | [📄 查看报告](experiments/exp01-bert-xnli/report.md) |
| **Exp 2** | GPT OELM | TinyStories | 分头正交实现成功，生成任务性能损失 -9.8% | [📄 查看报告](experiments/exp02-gpt-oelm/report.md) |
| **Exp 3** | GPT Ablation | TinyStories, OpenWebText, WikiText | 验证正交必要性，规模效应显著 | [📄 查看报告](experiments/exp03-gpt-ablation/report.md) |
| **Exp 4** | GPT Classification | IMDB, AG News, XNLI, MNLI | **分类任务 OELM 有效！平均 +8.14%** | [📄 查看报告](experiments/exp04-gpt-classification/report.md) |
| **Exp 5** | BERT OELM Paper | SST-2, MNLI | 论文级验证，正交必要性确认 | [📄 查看报告](experiments/exp05-bert-paper/report.md) |
| **Exp 6** | Multi-Dataset Validation | AG News, SST-2, XNLI, MNLI | **12/12 实验完成，平均 +10.46% 提升** | [📄 查看报告](experiments/exp06-multi-dataset/report.md) |

### 当前正式实验线

- `experiments/oelm-pretrain/`: 旧的 pilot/legacy 预训练验证目录，保留作参考。
- `experiments/oelm-pretrain-v2/`: 当前正式的 V2 预训练验证目录。
- V2 的原则是先做冻结逻辑和资源记录审计，再解释预训练与下游结果。
- V2 的所有正式 cluster 作业统一固定在 `cluster02 + gpu:pro6000`。

### 关键结论

> **任务类型决定 OELM 有效性，而非架构类型**

| 架构 | 任务类型 | OELM 效果 | 说明 |
|:----:|:---------|:---------:|:-----|
| BERT (Encoder) | 分类 | ✅ **有效** | +1.08%，训练快 57% |
| GPT (Decoder) | **分类** | ✅ **有效** | **平均 +8.14% ~ +10.46%** |
| GPT (Decoder) | 生成 | ❌ 无效 | 性能损失 -9.8% ~ -15.5% |

---

## 🗂️ 目录结构

```
Orthogonal-ELM-Transformers/
├── README.md                    # 本文件 - 项目总览
├── LICENSE                      # MIT 许可证
│
├── experiments/                 # ⭐ 实验目录
│   ├── exp01-bert-xnli/        # 实验1: BERT XNLI
│   │   ├── scripts/            # 运行脚本
│   │   ├── results/            # 实验结果
│   │   └── report.md           # 实验报告
│   │
│   ├── exp02-gpt-oelm/         # 实验2: GPT OELM
│   │   ├── scripts/
│   │   ├── results/
│   │   └── report.md
│   │
│   ├── exp03-gpt-ablation/     # 实验3: GPT 消融
│   │   ├── scripts/
│   │   ├── results/
│   │   └── report.md
│   │
│   ├── exp04-gpt-classification/  # 实验4: GPT 分类
│   │   ├── scripts/
│   │   ├── results/
│   │   └── report.md
│   │
│   ├── exp05-bert-paper/       # 实验5: BERT 论文实验
│   │   ├── scripts/
│   │   ├── results/
│   │   └── report.md
│   │
│   └── exp06-multi-dataset/    # 实验6: 多数据集验证
│       ├── scripts/
│       ├── results/
│       └── report.md
│
│   ├── oelm-pretrain/          # 旧 pilot 预训练验证目录
│   │   ├── scripts/
│   │   └── QUICKSTART.md
│   │
│   └── oelm-pretrain-v2/       # 当前正式 V2 预训练验证目录
│       ├── PLAN.md
│       ├── README.md
│       ├── configs/
│       ├── scripts/
│       ├── audits/
│       ├── reports/
│       └── manifests/
│
└── shared/                     # 共享代码（可选）
    └── models/                 # 共享模型定义
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/Starryyu77/Orthogonal-ELM-Transformers.git
cd Orthogonal-ELM-Transformers

# 安装依赖
pip install torch>=2.0.0 transformers datasets numpy scikit-learn tqdm
```

### 2. 运行实验

以 **Exp 6: 多数据集验证** 为例：

```bash
# 进入实验目录
cd experiments/exp06-multi-dataset

# 查看实验报告
cat report.md

# 运行 AG News 实验
cd scripts
sbatch run_agnews_baseline.sh   # Baseline
sbatch run_agnews_qk.sh         # OELM-QK
sbatch run_agnews_qk_ffn.sh     # OELM-QK-FFN
```

### 2.1 运行当前 V2 预训练验证

```bash
# 进入 V2 目录
cd experiments/oelm-pretrain-v2

# 查看 V2 计划
cat PLAN.md

# 查看 V2 使用说明
cat README.md

# Cluster 运行脚本位于 scripts/
ls scripts/run_phase*.sh
```

### 3. 查看结果

```bash
# 查看实验结果
cd ../results
cat ag_news/baseline/results.json
cat ag_news/oelm_qk_ffn/results.json
```

---

## 📈 核心结果摘要

### Exp 6: 多数据集验证（最新）

| 数据集 | 方法 | 准确率 | 训练时间 | 可训练参数 | vs Baseline |
|:-------|:-----|:-------|:---------|:-----------|:------------|
| **AG News** | Baseline | 92.46% | 2h04m | 38.54M (100%) | - |
| | **OELM-QK-FFN** | **92.32%** | **1h42m** | **25.05M (65%)** | **-0.14%** |
| **SST-2** | Baseline | 77.29% | 18m38s | 38.54M (100%) | - |
| | **OELM-QK-FFN** | **82.22%** | **10m08s** | **25.05M (65%)** | **+4.93%** 🏆 |
| **XNLI** | Baseline | 52.49% | 6h15m | 38.54M (100%) | - |
| | **OELM-QK-FFN** | **61.00%** | **5h25m** | **25.05M (65%)** | **+8.51%** 🏆 |
| **MNLI** | Baseline | 31.82% | 6h22m | 38.54M (100%) | - |
| | **OELM-QK-FFN** | **60.35%** | **5h28m** | **25.05M (65%)** | **+28.53%** 🏆 |

**总结**: 65% 参数，15% 更快，+10.46% 平均提升

---

## 🔑 使用建议

### 对于新的分类任务

```python
# 推荐：直接使用 OELM-QK-FFN
from shared.models import OELMForSequenceClassification

model = OELMForSequenceClassification(
    d_model=512,
    num_layers=6,
    num_heads=8,
    freeze_qk=True,      # 冻结 Q/K
    freeze_ffn=True,     # 冻结 FFN
    init_method='orthogonal'
)
```

### 预期效果

| 任务类型 | 预期提升 | 推荐方法 |
|:---------|:---------|:---------|
| 自然语言推理 (NLI) | +10% ~ +30% | OELM-QK-FFN |
| 情感分析 | +3% ~ +8% | OELM-QK-FFN |
| 主题分类 | -1% ~ +2% | OELM-QK-FFN |
| 文本生成 | 不推荐 | Baseline |

---

## 📚 相关资源

- **GitHub**: https://github.com/Starryyu77/Orthogonal-ELM-Transformers
- **实验详情**: 见各实验目录下的 `report.md`
- **服务器**: NTU EEE GPU Cluster (`10.97.216.128`)

---

## 📄 引用

如果本项目对您有帮助，请引用：

```bibtex
@article{oelm2025,
  title={Orthogonal ELM Transformers: Efficient Training via Q/K Freezing},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

**最后更新**: 2025年3月  
**版本**: v2.0 (重构版)
