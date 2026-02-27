# OELM Transformers 实验集合

> 所有 OELM (Orthogonal Extreme Learning Machine) Transformer 实验的统一入口

---

## 实验概览

本项目包含四个主要实验部分，验证OELM方法在不同架构和任务上的有效性：

| 实验 | 名称 | 状态 | 核心结果 | 详细报告 |
|------|------|------|----------|----------|
| **Phase 1** | BERT XNLI | ✅ 完成 | OELM优于Baseline (+1.08%)，训练快57% | [REPORT](./phase1-bert-xnli/REPORT.md) |
| **Phase 2** | GPT OELM | ✅ 完成 | 分头正交实现成功 | [REPORT](./phase2-gpt-oelm/REPORT.md) |
| **Phase 3** | GPT消融 | ✅ 100%完成 | 性能损失-9.8%~-15.5% | [REPORT](./phase3-gpt-ablation/REPORT.md) |
| **Paper** | BERT OELM论文 | ✅ 完成 | SST-2/MNLI，正交必要性验证 | [SUMMARY](./paper-bert-oelm/EXPERIMENT_SUMMARY.md) |
| **Phase 4** | GPT分类验证 | ⏸️ 计划中 | 验证是"任务类型"还是"架构"决定OELM效果 | [PLAN](./phase4-gpt-classification/PLAN.md) |

---

## 关键发现

### ✅ 成功
- **正交初始化有效**: OELM-Freeze比OELM-Random好6.0%
- **BERT上表现优秀**: XNLI上+1.08%，训练快57%；SST-2上达到98%性能

### ❌ 失败
- **GPT上效果不佳**: 性能损失-9.8%~-15.5%
- **无速度优势**: GPT上训练时间与Baseline相同
- **规模效应**: 数据集越大，性能损失越严重 (9.8% → 14.9% → 15.5%)

### 💡 核心洞察
| 架构 | 任务 | OELM适用性 | 原因 |
|------|------|------------|------|
| BERT (编码器) | 分类 | ✅ 适用 | 注意力模式稳定 |
| GPT (解码器) | 生成 | ❌ 不适用 | 需要动态Q/K调整 |

---

## 目录结构

```
experiments/
├── README.md                      # 本文件 - 实验总览
│
├── phase1-bert-xnli/              # Phase 1: BERT XNLI实验
│   ├── README.md                  # 实验说明
│   ├── REPORT.md                  # ⭐ 详细报告
│   ├── models/                    # 模型代码
│   │   ├── modeling_bert_oelm.py
│   │   └── train_bert.py
│   ├── scripts/                   # 启动脚本
│   ├── docs/                      # 文档
│   └── logs/                      # 日志
│
├── phase2-gpt-oelm/               # Phase 2: GPT OELM
│   ├── README.md
│   ├── REPORT.md                  # ⭐ 详细报告
│   ├── models/                    # 模型代码
│   │   ├── modeling_oelm_v2.py
│   │   ├── modeling_gpt.py
│   │   └── train_v2.py
│   ├── scripts/                   # 训练脚本
│   ├── data/                      # 数据准备
│   ├── docs/                      # 文档
│   ├── checkpoints/               # 检查点
│   └── outputs/                   # 实验输出
│
├── phase3-gpt-ablation/           # Phase 3: GPT消融
│   ├── README.md
│   ├── REPORT.md                  # ⭐ 详细报告
│   ├── PLAN.md                    # 实验计划
│   └── scripts/                   # 7个实验脚本
│       ├── run_gpt01.sh ... run_gpt07.sh
│
├── paper-bert-oelm/               # BERT OELM论文实验
│   ├── README.md                  # 完整项目README
│   ├── EXPERIMENT_SUMMARY.md      # 实验汇总
│   ├── src/                       # 源代码
│   ├── scripts/                   # 脚本
│   ├── configs/                   # 配置
│   ├── results/                   # 结果
│   └── docs/                      # 文档
│
├── phase4-gpt-classification/     # Phase 4: GPT分类任务验证 ⏸️
│   ├── PLAN.md                    # 实施计划
│   ├── models/                    # 模型代码（计划中）
│   ├── scripts/                   # 启动脚本（计划中）
│   └── data/                      # 数据准备（计划中）
│
└── common/                        # 共享工具
    └── scripts/
        ├── analyze_results.py
        └── monitor_experiments.sh
```

---

## 实验结果汇总

### BERT实验 (Phase 1 & Paper) ✅

| 数据集 | 任务 | Baseline | OELM-Freeze | 对比 |
|--------|------|----------|-------------|------|
| **XNLI** | 3分类NLI | 76.71% | **77.79%** | **+1.08%** ✅ |
| **SST-2** | 2分类情感 | 93.12% | 91.28% | -1.84% (达到98%) |
| **MNLI** | 3分类NLI | 83.44% | 82.23% | -1.21% (达到98.5%) |

**结论**: OELM在BERT分类任务上有效

→ [Phase 1报告](./phase1-bert-xnli/REPORT.md) | [Paper汇总](./paper-bert-oelm/EXPERIMENT_SUMMARY.md)

### GPT实验 (Phase 3) ✅

| ID | 数据集 | 方法 | PPL | 差距 | 状态 |
|----|--------|------|-----|------|------|
| GPT-01 | TinyStories | Baseline | 4.27 | - | ✅ |
| GPT-02 | TinyStories | OELM-Freeze | 4.69 | **+9.8%** ❌ | ✅ |
| GPT-03 | TinyStories | OELM-Random | 4.97 | +16.4% | ✅ |
| GPT-04 | OpenWebText | Baseline | 47.24 | - | ✅ |
| GPT-05 | OpenWebText | OELM-Freeze | 54.29 | **+14.9%** ❌ | ✅ |
| GPT-06 | WikiText-103 | Baseline | 25.13 | - | ✅ |
| GPT-07 | WikiText-103 | OELM-Freeze | 29.03 | **+15.5%** ❌ | ✅ |

**结论**: 所有数据集都超出5%目标，且规模越大损失越严重

→ [Phase 3报告](./phase3-gpt-ablation/REPORT.md)

---

## 快速开始

### 启动实验

```bash
# Phase 1: BERT XNLI
cd phase1-bert-xnli/scripts
./run_xnli_experiments.sh

# Phase 2: GPT OELM
cd phase2-gpt-oelm/scripts
./run_phase2_experiments.sh

# Phase 3: 消融实验
cd phase3-gpt-ablation/scripts
./run_gpt01.sh 2  # GPU 2
./run_gpt02.sh 3  # GPU 3

# Paper: BERT OELM
cd paper-bert-oelm
python src/train_bert.py --freeze_mode true --init_method orthogonal
```

### 监控实验

```bash
# 监控所有实验
cd common/scripts
./monitor_experiments.sh

# 实时刷新
./monitor_experiments.sh live
```

---

## 对比分析

### BERT vs GPT

| 维度 | BERT | GPT |
|------|------|-----|
| 任务 | 分类 | 生成 |
| 最佳结果 | +1.08% ✅ | -9.8% ❌ |
| 速度提升 | 57% ✅ | 0% ❌ |
| 参数节省 | 12.9% | 12.9% |
| 目标达成 | ✅ | ❌ |

### 消融分析

| 数据集 | Baseline | OELM-Freeze | OELM-Random | 正交价值 |
|--------|----------|-------------|-------------|----------|
| TinyStories | 4.27 | 4.69 ❌ | 4.97 | +6.0% ✅ |
| OpenWebText | 47.24 | 54.29 ❌ | - | - |
| WikiText-103 | 25.13 | 29.03 ❌ | - | - |

---

## 相关文档

- [根目录README.md](../README.md) - 项目主入口
- [最终实验报告](../docs/FINAL_EXPERIMENT_REPORT.md) - 完整实验结果汇总
- [Phase 1报告](./phase1-bert-xnli/REPORT.md) - BERT XNLI详细报告
- [Phase 2报告](./phase2-gpt-oelm/REPORT.md) - GPT OELM移植报告
- [Phase 3报告](./phase3-gpt-ablation/REPORT.md) - GPT消融实验报告
- [Paper汇总](./paper-bert-oelm/EXPERIMENT_SUMMARY.md) - BERT论文实验汇总
- [Phase 4计划](./phase4-gpt-classification/PLAN.md) - GPT分类验证实验计划

---

**最后更新**: 2026-02-12 (添加Phase 4计划)
