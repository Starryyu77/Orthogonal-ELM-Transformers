# Orthogonal ELM Transformers

> 基于分头正交初始化(Head-wise Orthogonal Initialization)的Transformer高效训练研究

---

## 项目简介

本项目探索在Transformer架构中使用**分头正交初始化**配合**冻结Q/K参数**的方法，旨在减少可训练参数数量同时保持模型性能。

### 核心创新

1. **分头正交初始化**: 每个注意力头独立进行QR分解，保持head内部几何结构
2. **冻结Q/K参数**: 训练过程中冻结Query/Key投影，只训练Value/Output
3. **跨架构验证**: 在BERT(编码器)和GPT(解码器)上分别验证

---

## 快速导航

### 实验阶段

| 阶段 | 名称 | 状态 | 关键结果 | 链接 |
|------|------|------|----------|------|
| **Phase 1** | BERT XNLI | ✅ 完成 | OELM优于Baseline (+1.08%)，训练快57% | [`experiments/phase1-bert-xnli/`](./experiments/phase1-bert-xnli/) |
| **Phase 2** | GPT OELM | ✅ 完成 | 分头正交实现成功 | [`experiments/phase2-gpt-oelm/`](./experiments/phase2-gpt-oelm/) |
| **Phase 3** | GPT消融 | ✅ 100%完成 | 生成任务性能损失-9.8%~-15.5% | [`experiments/phase3-gpt-ablation/`](./experiments/phase3-gpt-ablation/) |
| **Phase 4** | GPT分类验证 | ✅ 完成 | **分类任务OELM有效！平均+8.14%** | [`experiments/phase4-gpt-classification/`](./experiments/phase4-gpt-classification/) |
| **Paper** | BERT OELM论文 | ✅ 完成 | SST-2/MNLI实验，正交必要性验证 | [`experiments/paper-bert-oelm/`](./experiments/paper-bert-oelm/) |

### 重要文档

- **[最终实验报告](./docs/FINAL_EXPERIMENT_REPORT.md)** - 完整实验结果汇总与分析
- **[实验总览](./experiments/README.md)** - 所有实验的完整归档
- **[Phase 3日志](./EXPERIMENT_LOG_Phase3.md)** - 详细实验日志

---

## 项目结构

```
Train/
├── README.md                      # ⭐ 本文件 - 项目主入口
├── EXPERIMENTS_COMPLETE.md        # 实验总览
├── EXPERIMENT_STATUS.md           # 当前状态
│
├── experiments/                   # ⭐ 实验目录（推荐入口）
│   ├── README.md                  # 实验总览
│   │
│   ├── phase1-bert-xnli/          # Phase 1: BERT XNLI实验
│   │   ├── README.md              # 实验说明
│   │   ├── REPORT.md              # 详细报告
│   │   ├── models/                # 模型代码
│   │   ├── scripts/               # 启动脚本
│   │   ├── docs/                  # 文档
│   │   └── logs/                  # 日志
│   │
│   ├── phase2-gpt-oelm/           # Phase 2: GPT OELM
│   │   ├── README.md
│   │   ├── REPORT.md
│   │   ├── models/                # 模型代码
│   │   ├── scripts/               # 训练脚本
│   │   ├── data/                  # 数据准备
│   │   ├── docs/                  # 文档
│   │   ├── checkpoints/           # 检查点
│   │   └── outputs/               # 实验输出
│   │
│   ├── phase3-gpt-ablation/       # Phase 3: GPT消融实验
│   │   ├── README.md
│   │   ├── REPORT.md
│   │   ├── PLAN.md                # 实验计划
│   │   └── scripts/               # 7个实验脚本
│   │
│   ├── paper-bert-oelm/           # BERT OELM论文实验
│   │   ├── README.md              # 已有完整README
│   │   ├── EXPERIMENT_SUMMARY.md  # 实验汇总
│   │   ├── src/                   # 源代码
│   │   ├── scripts/               # 脚本
│   │   ├── configs/               # 配置
│   │   ├── results/               # 结果
│   │   └── docs/                  # 文档
│   │
│   └── common/                    # 共享工具
│       └── scripts/
│           ├── analyze_results.py
│           └── monitor_experiments.sh
│
├── docs/                          # 项目文档
├── tools/                         # 工具脚本
│   └── cluster_setup/
└── archive/                       # 归档文件
```

---

## 关键发现

### ✅ BERT上表现优秀 (Phase 1 & Paper)

| 指标 | Baseline | OELM-Freeze | 对比 |
|------|----------|-------------|------|
| 准确率 (XNLI) | 76.71% | **77.79%** | **+1.08%** ✅ |
| 准确率 (SST-2) | 93.12% | 91.28% | -1.84% (达到98%) |
| 每步时间 | 0.162s | 0.069s | **-57.2%** ⭐ |
| 参数节省 | - | 12.9% | 14.2M参数冻结 |

**结论**: 冻结Q/K在**分类任务**上非常有效

### ✅ GPT分类任务表现出色 (Phase 4)

| 数据集 | 类别数 | Baseline | OELM-Freeze | 提升 |
|--------|--------|----------|-------------|------|
| **IMDB** | 2 | 78.56% | **85.70%** | **+7.14%** ✅ |
| **AG News** | 4 | 87.05% | **92.74%** | **+5.69%** ✅ |
| **XNLI** | 3 | 46.39% | **57.99%** | **+11.60%** ✅ |
| **平均** | - | - | - | **+8.14%** ✅ |

**结论**: 冻结Q/K在**分类任务**上非常有效，无论BERT还是GPT！

### ❌ GPT生成任务效果不佳 (Phase 2 & 3)

| 数据集 | Baseline PPL | OELM-Freeze PPL | 性能损失 |
|--------|-------------|-----------------|----------|
| TinyStories | 4.27 | 4.69 | **-9.8%** ❌ |
| OpenWebText | 47.24 | 54.29 | **-14.9%** ❌ |
| WikiText-103 | 25.13 | 29.03 | **-15.5%** ❌ |

**结论**: 冻结Q/K在**生成任务**上代价过大

### 💡 核心洞察：任务类型决定论

| 架构 | 任务类型 | OELM效果 | 结论 |
|------|----------|----------|------|
| BERT (编码器) | 分类 | ✅ **优于Baseline** | OELM有效 |
| GPT (解码器) | **分类** | ✅ **优于Baseline** | **OELM有效！** |
| GPT (解码器) | 生成 | ❌ 劣于Baseline | OELM无效 |

> **关键发现**: 不是架构问题，是任务类型问题！
> 分类任务适合OELM，生成任务不适合，无论Encoder还是Decoder架构。

---

## 使用方法

### 启动实验

```bash
# Phase 1: BERT XNLI
cd experiments/phase1-bert-xnli/scripts
./run_xnli_experiments.sh

# Phase 2: GPT OELM
cd experiments/phase2-gpt-oelm/scripts
./run_phase2_experiments.sh

# Phase 3: 消融实验
cd experiments/phase3-gpt-ablation/scripts
./run_gpt01.sh 2  # TinyStories Baseline on GPU 2
./run_gpt02.sh 3  # TinyStories OELM on GPU 3

# Phase 4: GPT分类验证
cd experiments/phase4-gpt-classification/scripts
./run_imdb_baseline.sh 0
./run_imdb_oelm.sh 1
./run_agnews_baseline.sh 0
./run_agnews_oelm.sh 1
./run_xnli_baseline.sh 0
./run_xnli_oelm.sh 1

# Paper: BERT OELM
cd experiments/paper-bert-oelm
python src/train_bert.py --freeze_mode true --init_method orthogonal
```

### 监控实验

```bash
# 查看所有实验状态
./experiments/common/scripts/monitor_experiments.sh

# 实时监控
./experiments/common/scripts/monitor_experiments.sh live
```

---

## 服务器信息

- **地址**: `10.97.216.128`
- **用户名**: `tianyu016`
- **项目路径**: `/projects/Orthogonal_ELM_Transformers/Train`
- **GPU**: 4x RTX A5000

```bash
# 连接服务器
ssh tianyu016@10.97.216.128
```

---

## 项目总结

### 完成状态

🎉 **所有实验已完成！**

- **Phase 1**: BERT XNLI - OELM 优于 Baseline (+1.08%)，训练快 57%
- **Phase 2**: GPT OELM 移植 - 分头正交实现成功
- **Phase 3**: GPT 消融实验 (7/7) - 性能损失 -9.8%~-15.5%
- **Paper**: BERT SST-2/MNLI - 达到 98%+ 性能

### 核心结论：任务类型决定论

| 任务类型 | 架构 | OELM 效果 | 说明 |
|----------|------|-----------|------|
| **分类** | BERT (编码器) | ✅ **有效** | +1.08%，训练快57% |
| **分类** | GPT (解码器) | ✅ **有效** | **平均+8.14%，速度更快** |
| 生成 | GPT (解码器) | ❌ **无效** | 性能损失 9.8%~15.5% |

**核心发现**: 任务类型决定OELM有效性，而非架构类型！
- 分类任务：适合OELM（双向attention，固定表示空间）
- 生成任务：不适合OELM（因果依赖，动态表示空间）

---

## 详细报告

- **[Phase 4 完整报告](./experiments/phase4-gpt-classification/REPORT.md)** - GPT分类实验详细分析
- **[实验总览](./docs/EXPERIMENTS.md)** - 所有实验完整归档
- **[最终实验报告](./docs/FINAL_EXPERIMENT_REPORT.md)** - 历史实验结果汇总

---

**最后更新**: 2026-02-12
