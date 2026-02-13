# OELM Transformers 实验文档

## 实验总览

本项目系统性地验证了 Head-wise Orthogonal ELM Transformer (OELM) 在不同架构和任务上的有效性。

## 核心发现：任务类型决定论

```
分类任务 + OELM = ✅ 有效 (无论Encoder/Decoder)
生成任务 + OELM = ❌ 无效 (无论Encoder/Decoder)
```

## 实验阶段

### Phase 1: BERT XNLI 分类实验 ✅
- **目标**: 验证OELM在Encoder架构+分类任务上的效果
- **结果**: OELM比Baseline高 **+1.08%**，训练速度 **+57%**
- **状态**: 已完成
- **位置**: `experiments/phase1-bert-xnli/`

### Phase 2: GPT 语言建模实验 ✅
- **目标**: 验证OELM在Decoder架构+生成任务上的效果
- **结果**: OELM性能下降 **-9.8%~-15.5%**
- **状态**: 已完成
- **位置**: `experiments/phase2-gpt-oelm/`

### Phase 3: GPT 消融实验 ✅
- **目标**: 分析OELM失效原因
- **结果**: 正交初始化有价值（比随机好6%），但冻结Q/K在生成任务上无效
- **状态**: 已完成
- **位置**: `experiments/phase3-gpt-ablation/`

### Phase 4: GPT 分类实验 ✅ **(最新完成)**
- **目标**: 验证OELM在Decoder架构+分类任务上的效果
- **结果**: **OELM consistently 有效，平均提升 8.14%**
- **状态**: 已完成，所有结果已同步到本地
- **位置**: `experiments/phase4-gpt-classification/`
- **完整报告**: [REPORT.md](../experiments/phase4-gpt-classification/REPORT.md)

---

## Phase 4 详细结果

### 实验数据对比

| 数据集 | 类别数 | Baseline | OELM-Freeze | 绝对提升 | 相对提升 | 速度提升 |
|--------|--------|----------|-------------|----------|----------|----------|
| **IMDB** | 2 | 78.56% | **85.70%** | **+7.14%** | +9.1% | -3.6% |
| **AG News** | 4 | 87.05% | **92.74%** | **+5.69%** | +6.5% | -5.7% |
| **XNLI** | 3 | 46.39% | **57.99%** | **+11.60%** | +25.0% | **-21.0%** |
| **平均** | - | - | - | **+8.14%** | **+13.5%** | **-10.1%** |

### 完整实验矩阵

| 实验 | 架构 | 任务 | Baseline | OELM | 效果 |
|------|------|------|----------|------|------|
| BERT XNLI | Encoder | 分类 | - | - | **+1.08%** ✅ |
| GPT IMDB | Decoder | 分类 | 78.56% | 85.70% | **+7.14%** ✅ |
| GPT AG News | Decoder | 分类 | 87.05% | 92.74% | **+5.69%** ✅ |
| GPT XNLI | Decoder | 分类 | 46.39% | 57.99% | **+11.60%** ✅ |
| GPT TinyStories | Decoder | 生成 | - | - | **-9.8%** ❌ |
| GPT OpenWebText | Decoder | 生成 | - | - | **-14.9%** ❌ |

---

## 关键洞察

### 1. 任务类型 > 架构类型
- **分类任务**（无论BERT/GPT）：OELM有效
- **生成任务**（无论BERT/GPT）：OELM无效

### 2. 为什么分类任务适合OELM？
- 双向attention，与Encoder类似
- 固定表示空间，正交初始化提供良好初始点
- 池化机制减少对因果依赖的需求

### 3. 为什么生成任务不适合OELM？
- 严格的因果attention依赖
- 需要动态变化的表示空间
- 自回归特性不适应正交初始化

---

## 快速开始

### 环境配置

```bash
# 安装依赖
pip install torch datasets transformers
```

### 运行Phase 4实验

```bash
cd experiments/phase4-gpt-classification

# IMDB
./scripts/run_imdb_baseline.sh 0
./scripts/run_imdb_oelm.sh 1

# AG News
./scripts/run_agnews_baseline.sh 0
./scripts/run_agnews_oelm.sh 1

# XNLI
./scripts/run_xnli_baseline.sh 0
./scripts/run_xnli_oelm.sh 1
```

---

## 结果位置

### 集群结果
```
ntu-gpu43:~/Orthogonal_ELM_Transformers/Train/outputs/
├── IMDB_baseline/          ✅
├── IMDB_oelm_freeze/       ✅
├── AGNews_baseline/        ✅
├── AGNews_oelm_freeze/     ✅
├── XNLI_baseline/          ✅
└── XNLI_oelm_freeze/       ✅
```

### 本地同步结果
```
./outputs_phase4/
├── IMDB_baseline/          ✅
├── IMDB_oelm_freeze/       ✅
├── AGNews_baseline/        ✅
├── AGNews_oelm_freeze/     ✅
├── XNLI_baseline/          ✅
└── XNLI_oelm_freeze/       ✅
```

---

## 监控命令

```bash
# 查看日志
ssh ntu-gpu43 'tail -f ~/Orthogonal_ELM_Transformers/Train/outputs/XNLI_baseline/console.log'

# GPU状态
ssh ntu-gpu43 'nvidia-smi'

# 查看所有tmux会话
ssh ntu-gpu43 'tmux ls'
```

---

## 相关文档

- [Phase 4 完整报告](../experiments/phase4-gpt-classification/REPORT.md)
- [运行手册](RUNBOOK.md) - 详细的实验执行和故障排查指南
- [EXPERIMENTS_COMPLETE.md](../EXPERIMENTS_COMPLETE.md) - 归档文档

---

*最后更新: 2026-02-13*
