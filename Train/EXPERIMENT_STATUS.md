# 实验执行状态总览

> 更新时间: 2026-02-12
> 当前阶段: Phase 3 (71% 完成)

---

## Phase 1: XNLI 实验 (BERT) ✅ 已完成

### 最终结果

| 指标 | Baseline | OELM-Freeze | 对比 |
|------|----------|-------------|------|
| **最佳准确率** | 76.71% | **77.79%** | OELM **+1.08%** ✅ |
| 可训练参数 | 109.5M (100%) | 95.3M (87.1%) | 节省 12.9% |
| 纯训练时间 | 11,971s | 5,138s | **-57.1%** ⭐ |
| 每步时间 | 0.162s | 0.069s | **-57.2%** ⭐ |

### 结论
- ✅ **OELM在BERT上表现优秀**: 准确率提升+1.08%，训练速度提升57%
- ✅ **冻结Q/K在分类任务上有效**

---

## Phase 2: GPT OELM 移植 ✅ 已完成

### 已完成工作

#### 1. 核心模型 (✅ 完成)
- **文件**: `gpt-oelm-project/models/modeling_oelm_v2.py`
- **关键修正**: 从全局正交改为分头正交初始化
- **创新点**: 每个 head 独立 QR 分解

#### 2. 训练脚本 (✅ 完成)
- **文件**: `gpt-oelm-project/scripts/train_v2.py`
- **功能**: 支持 `baseline`/`oelm_v2`/`oelm_random` 三种模式

---

## Phase 3: GPT 消融实验 🟢 进行中 (5/7 完成)

### 实验矩阵

| ID | 数据集 | 方法 | PPL | 与Baseline差距 | 状态 | 训练时间 |
|----|--------|------|-----|----------------|------|----------|
| GPT-01 | TinyStories | Baseline | 4.27 | - | ✅ 完成 | 4h 31m |
| GPT-02 | TinyStories | OELM-Freeze | 4.69 | **+9.8%** | ✅ 完成 | 4h 32m |
| GPT-03 | TinyStories | OELM-Random | 4.97 | +16.4% | ✅ 完成 | 4h 15m |
| GPT-04 | OpenWebText | Baseline | 47.24 | - | ✅ 完成 | 9h 10m |
| GPT-05 | OpenWebText | OELM-Freeze | 54.29 | **+14.9%** | ✅ 完成 | 9h 8m |
| GPT-06 | WikiText-103 | Baseline | TBD | - | ⏳ 待启动 | - |
| GPT-07 | WikiText-103 | OELM-Freeze | TBD | TBD | ⏳ 待启动 | - |

### 关键发现

#### 1. 正交初始化有效 ✅
- OELM-Freeze (PPL 4.69) 比 OELM-Random (PPL 4.97) 好 **6.0%**
- 证明正交初始化本身有价值

#### 2. 冻结Q/K在GPT上代价大 ❌
- TinyStories: +9.8% (超出5%目标)
- OpenWebText: +14.9% (超出5%目标)
- **目标未达成**: 所有数据集都超出5%性能损失目标

#### 3. 数据集越大，性能损失越大 ⚠️
- OpenWebText (+14.9%) > TinyStories (+9.8%)
- 说明复杂任务对Q/K冻结更敏感

#### 4. 无训练速度优势 ❌
- 训练时间与Baseline相同 (~0.184s/步)
- 与BERT实验不同（BERT有57%速度提升）

---

## 脚本完整性

### 实验启动脚本
```
experiments/phase3-gpt-ablation/scripts/
├── run_gpt01.sh  # ✅ TinyStories Baseline
├── run_gpt02.sh  # ✅ TinyStories OELM-Freeze
├── run_gpt03.sh  # ✅ TinyStories OELM-Random (消融)
├── run_gpt04.sh  # ✅ OpenWebText Baseline
├── run_gpt05.sh  # ✅ OpenWebText OELM-Freeze
├── run_gpt06.sh  # ✅ WikiText-103 Baseline (待启动)
└── run_gpt07.sh  # ✅ WikiText-103 OELM-Freeze (待启动)
```

### 核心代码
```
gpt-oelm-project/
├── models/modeling_oelm_v2.py  # ✅ 分头正交实现
├── scripts/train_v2.py         # ✅ 主训练脚本
└── outputs/                    # ✅ 5个已完成实验数据
```

---

## 下一步行动

### 短期 (本周)
1. ⏳ **WikiText-103数据准备** (如需要)
2. ⏳ **启动 GPT-06/07** (如需要)
3. ⏳ **完成Phase 3所有实验**

### 中期 (下周)
1. 📊 **全面结果分析**
2. 🔬 **设计改进策略** (部分解冻、分层学习率)
3. 📝 **撰写完整实验报告**

### 可能的改进方向
- **部分解冻**: 只解冻最后1-2层的Q/K
- **分层学习率**: 可训练参数使用更高学习率
- **渐进式解冻**: 训练过程中逐渐解冻Q/K

---

## 文件清单

### 实验数据
```
gpt-oelm-project/outputs/
├── GPT-01_baseline/              # ✅ 完成
├── GPT-02_oelm_freeze/           # ✅ 完成
├── GPT-03_oelm_random/           # ✅ 完成
├── GPT-04_openwebtext_baseline/  # ✅ 完成
└── GPT-05_openwebtext_oelm/      # ✅ 完成
```

### 相关文档
- [EXPERIMENTS_COMPLETE.md](./EXPERIMENTS_COMPLETE.md) - 完整归档
- [EXPERIMENT_LOG_Phase3.md](./EXPERIMENT_LOG_Phase3.md) - Phase 3详细日志
- [EXPERIMENT_PLAN_v2.md](./EXPERIMENT_PLAN_v2.md) - 实验规划

---

**备注**: Phase 1 (BERT/XNLI) 和 Phase 2 (GPT移植) 已完成。Phase 3 (GPT消融) 71%完成。
