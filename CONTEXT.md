# Orthogonal ELM Transformers - 项目状态记录

**最后更新**: 2025年3月19日
**当前状态**: ✅ 仓库重构完成

---

## 🎉 完成的工作

### 1. 仓库完全重构

**旧状态**: Train/ 目录混乱，包含 255+ 个混乱文件
**新状态**: 清晰的 experiments/ 结构，6 个独立实验

**执行的清理**:
- ✅ 删除 Train/ 目录（255 个文件，-88,580 行）
- ✅ 创建新的 README.md（清晰的实验导航）
- ✅ 创建 experiments/ 目录结构
- ✅ 移动核心模型代码到 shared/models/
- ✅ GitHub 提交: `b121aa3`

---

## 📁 当前目录结构

```
Orthogonal-ELM-Transformers/
├── README.md                    # 项目总览和实验导航
├── LICENSE                      # MIT 许可证
│
├── experiments/                 # 实验目录
│   ├── exp01-bert-xnli/        # BERT XNLI 实验
│   │   ├── scripts/            # 运行脚本
│   │   └── report.md           # 实验报告
│   │
│   ├── exp02-gpt-oelm/         # GPT OELM 实验
│   │   ├── scripts/
│   │   └── report.md
│   │
│   ├── exp03-gpt-ablation/     # GPT 消融实验
│   │   ├── scripts/
│   │   └── report.md
│   │
│   ├── exp04-gpt-classification/  # GPT 分类实验
│   │   ├── scripts/
│   │   └── report.md
│   │
│   ├── exp05-bert-paper/       # BERT 论文实验
│   │   ├── scripts/
│   │   └── report.md
│   │
│   ├── exp06-multi-dataset/    # 多数据集验证
│   │   ├── scripts/
│   │   └── report.md
│   │
│   └── shared/                 # 共享代码
│       └── models/             # 核心模型实现
│           ├── modeling_oelm_v2.py
│           ├── modeling_oelm_classification.py
│           ├── modeling_oelm_pretrain.py
│           ├── modeling_gpt.py
│           └── modeling_gpt_classification.py
│
└── .git/                       # Git 仓库
```

---

## 📊 Phase 6 实验结果（已完成）

**状态**: 12/12 完成 (100%)
**提交**: 基于预训练模型的下游任务微调

### 完整结果

| 数据集 | 方法 | 准确率 | 训练时间 | 可训练参数 | vs Baseline |
|:-------|:-----|:-------|:---------|:-----------|:------------|
| **AG News** | Baseline | 92.46% | 2h04m | 38.54M (100%) | - |
| | OELM-QK | 91.86% | 1h54m | 29.21M (75.8%) | -0.60% |
| | **OELM-QK-FFN** | **92.32%** | **1h42m** | **25.05M (65.0%)** | **-0.14%** |
| **SST-2** | Baseline | 77.29% | 18m38s | 38.54M (100%) | - |
| | OELM-QK | 78.78% | 16m28s | 29.21M (75.8%) | +1.49% |
| | **OELM-QK-FFN** | **82.22%** | **10m08s** | **25.05M (65.0%)** | **+4.93%** 🏆 |
| **XNLI** | Baseline | 52.49% | 6h15m | 38.54M (100%) | - |
| | OELM-QK | 33.33% | 6h18m | 29.21M (75.8%) | -19.16% |
| | **OELM-QK-FFN** | **61.00%** | **5h25m** | **25.05M (65.0%)** | **+8.51%** 🏆 |
| **MNLI** | Baseline | 31.82% | 6h22m | 38.54M (100%) | - |
| | **OELM-QK** | **58.25%** | **6h13m** | **29.21M (75.8%)** | **+26.43%** 🏆 |
| | **OELM-QK-FFN** | **60.35%** | **5h28m** | **25.05M (65.0%)** | **+28.53%** 🏆 |

**关键指标**:
- 平均准确率提升: **+10.46%**
- 平均训练时间节省: **~15%**
- 参数量减少: **35%** (从 38.54M 到 25.05M)

---

## 🔑 核心结论

### 1. OELM-QK-FFN 整体最优
- 3/4 数据集优于 Baseline
- 平均提升 +10.46%
- 仅用 65% 可训练参数

### 2. 任务类型显著影响效果
| 任务类型 | 效果 | 原因分析 |
|:---------|:-----|:---------|
| **NLI (MNLI, XNLI)** | 🔥 巨大提升 (+8% ~ +28%) | 复杂推理需要稳定的注意力模式 |
| **情感分析 (SST-2)** | ✅ 显著提升 (+4.93%) | 情感理解需要捕捉关键特征 |
| **简单分类 (AG News)** | ➡️ 基本持平 (-0.14%) | 任务简单，正则化效果不明显 |

### 3. 冻结策略至关重要
- 仅冻结 Q/K (OELM-QK): 在 XNLI 上失败
- 冻结 Q/K + FFN (OELM-QK-FFN): 全部成功
- FFN 冻结提供额外的正则化效果

---

## 📝 实验说明

**Phase 6 实验性质**: 基于预训练模型的下游任务微调

**实验流程**:
1. **Phase 5**: 预训练 (已完成)
   - 在通用语料上预训练 3 种模型
   - Baseline (100%), OELM-QK (75.8%), OELM-QK-FFN (65.0%)

2. **Phase 6**: 下游微调 (已完成)
   - 使用 Phase 5 预训练权重
   - 在 4 个分类数据集上微调
   - 对比不同冻结策略的效果

**训练时间含义**: 微调时间 (fine-tuning time)，不是从头训练

---

## 🔗 相关链接

- **GitHub**: https://github.com/Starryyu77/Orthogonal-ELM-Transformers
- **最新提交**: `b121aa3` - refactor: Complete repository cleanup
- **服务器**: NTU EEE GPU Cluster (`10.97.216.128`)

---

## ✅ 待办清单（已完成）

- [x] 分析现有实验结构
- [x] 创建新的根目录 README.md
- [x] 创建 experiments/ 目录结构
- [x] 整理 Phase 1-6 实验
- [x] 检查并转移重要内容
- [x] 删除旧的 Train/ 目录
- [x] 提交清理后的代码到 GitHub

---

**下一步**: 如需进一步调整实验报告或添加新实验，可直接在对应 experiments/expXX/ 目录下操作
