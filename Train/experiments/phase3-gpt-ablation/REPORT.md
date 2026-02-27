# Phase 3 实验报告: GPT 消融实验

> 在多个数据集上全面评估 OELM 方法的有效性

---

## 1. 实验目标

1. **多数据集验证**: TinyStories, OpenWebText, WikiText-103
2. **消融研究**: Baseline vs OELM-Freeze vs OELM-Random
3. **量化性能损失**: 精确测量OELM与Baseline的差距
4. **验证正交必要性**: OELM-Random vs OELM-Freeze对比

---

## 2. 实验矩阵

### 2.1 已完成实验 (5/7)

| ID | 数据集 | 方法 | PPL | 与Baseline差距 | 状态 | 训练时间 |
|----|--------|------|-----|----------------|------|----------|
| GPT-01 | TinyStories | Baseline | 4.27 | - | ✅ 完成 | 4h 31m |
| GPT-02 | TinyStories | OELM-Freeze | 4.69 | **+9.8%** ❌ | ✅ 完成 | 4h 32m |
| GPT-03 | TinyStories | OELM-Random | 4.97 | +16.4% | ✅ 完成 | 4h 15m |
| GPT-04 | OpenWebText | Baseline | 47.24 | - | ✅ 完成 | 9h 10m |
| GPT-05 | OpenWebText | OELM-Freeze | 54.29 | **+14.9%** ❌ | ✅ 完成 | 9h 8m |

### 2.2 待完成实验 (2/7)

| ID | 数据集 | 方法 | 状态 | 预计时间 |
|----|--------|------|------|----------|
| GPT-06 | WikiText-103 | Baseline | ⏳ 待启动 | ~12h |
| GPT-07 | WikiText-103 | OELM-Freeze | ⏳ 待启动 | ~12h |

---

## 3. 详细结果

### 3.1 TinyStories (短篇故事)

| 实验 | PPL | 与Baseline差距 | 训练时间 | 每步时间 |
|------|-----|----------------|----------|----------|
| **Baseline** | 4.27 | - | 4h 31m | 0.184s |
| **OELM-Freeze** | 4.69 | **+9.8%** ❌ | 4h 32m | 0.184s |
| **OELM-Random** | 4.97 | +16.4% | 4h 15m | 0.184s |

**分析**:
- 目标: PPL ≤ 4.48 (4.27 × 1.05)
- 实际: 4.69 (超出目标)
- 正交价值: OELM-Freeze比OELM-Random好6.0%

### 3.2 OpenWebText (网页文本)

| 实验 | PPL | 与Baseline差距 | 训练时间 | 每步时间 |
|------|-----|----------------|----------|----------|
| **Baseline** | 47.24 | - | 9h 10m | 0.184s |
| **OELM-Freeze** | 54.29 | **+14.9%** ❌ | 9h 8m | 0.184s |

**分析**:
- 复杂数据集上性能损失更大
- 9.8% → 14.9%，差距扩大

### 3.3 跨数据集趋势

```
数据集复杂度:    TinyStories  <  OpenWebText  <  WikiText-103
性能损失:        9.8%         <  14.9%         <  TBD

结论: 数据集越复杂，OELM性能损失越大
```

---

## 4. 消融分析

### 4.1 正交初始化必要性验证

| 对比 | PPL | 差距 |
|------|-----|------|
| OELM-Freeze (正交) | 4.69 | - |
| OELM-Random (随机) | 4.97 | +6.0% |

**结论**: ✅ 正交初始化有效
- OELM-Freeze比OELM-Random好6.0%
- 证明正交性本身有价值

### 4.2 冻结Q/K代价分析

| 对比 | PPL | 差距 |
|------|-----|------|
| Baseline (可训练) | 4.27 | - |
| OELM-Freeze (冻结) | 4.69 | **+9.8%** ❌ |

**结论**: ❌ 冻结Q/K代价大
- 性能损失9.8%，超出5%目标
- GPT生成任务对Q/K敏感

### 4.3 综合结论

```
正交初始化:  ✅ 有效 (+6.0% vs 随机)
冻结Q/K:     ❌ 代价大 (-9.8% vs Baseline)
整体效果:    ❌ 未达目标
```

---

## 5. 训练效率分析

### 5.1 时间效率

| 指标 | Baseline | OELM-Freeze | 对比 |
|------|----------|-------------|------|
| 总训练时间 | 4h 31m | 4h 32m | ~0% |
| 每步时间 | 0.184s | 0.184s | ~0% |
| 验证时间 | 相同 | 相同 | - |

**结论**: ❌ 无速度优势
- 与BERT不同（BERT快57%）
- GPT上冻结Q/K没有加速效果

### 5.2 参数效率

| 模型 | 可训练参数 | 节省比例 |
|------|-----------|----------|
| Baseline | 82M (100%) | - |
| OELM-Freeze | 71M (87.1%) | 12.9% |

**结论**: ✅ 参数节省12.9%
- 但未带来速度提升
- 可能原因: GPU并行计算，参数减少不影响速度

---

## 6. 对比总结

### 6.1 BERT vs GPT

| 维度 | BERT (分类) | GPT (生成) |
|------|-------------|------------|
| **准确率/PPL** | ✅ +1.08% | ❌ -9.8%~-14.9% |
| **训练速度** | ✅ 快57% | ❌ 无提升 |
| **参数效率** | ✅ 省12.9% | ✅ 省12.9% |
| **目标达成** | ✅ 达成 | ❌ 未达成 |

### 6.2 关键洞察

| 架构 | 任务 | OELM适用性 | 原因 |
|------|------|------------|------|
| BERT (编码器) | 分类 | ✅ 适用 | 注意力模式稳定，可预训练固定 |
| GPT (解码器) | 生成 | ❌ 不适用 | 需要动态调整Q/K，上下文变化大 |

---

## 7. 实验脚本

### 7.1 启动命令

```bash
cd experiments/phase3-gpt-ablation/scripts

# TinyStories
./run_gpt01.sh 2  # Baseline on GPU 2
./run_gpt02.sh 3  # OELM-Freeze on GPU 3
./run_gpt03.sh 2  # OELM-Random on GPU 2

# OpenWebText
./run_gpt04.sh 2  # Baseline on GPU 2
./run_gpt05.sh 3  # OELM-Freeze on GPU 3

# WikiText-103 (待启动)
./run_gpt06.sh 2  # Baseline on GPU 2
./run_gpt07.sh 3  # OELM-Freeze on GPU 3
```

### 7.2 关键脚本

| 脚本 | 用途 |
|------|------|
| `run_gpt01.sh` - `run_gpt07.sh` | 7个实验的启动脚本 |
| `../common/scripts/monitor_experiments.sh` | 实验监控 |

---

## 8. 相关文件

### 实验计划
- [`PLAN.md`](./PLAN.md) - 详细实验规划

### 代码位置
```
gpt-oelm-project/
├── models/
│   ├── modeling_oelm_v2.py       # OELM模型
│   ├── modeling_gpt.py           # Baseline模型
│   └── train_v2.py               # 训练脚本
└── outputs/
    ├── GPT-01_baseline/          # ✅
    ├── GPT-02_oelm_freeze/       # ✅
    ├── GPT-03_oelm_random/       # ✅
    ├── GPT-04_openwebtext_baseline/  # ✅
    └── GPT-05_openwebtext_oelm/      # ✅
```

---

## 9. 结论

### 9.1 主要结论

| 问题 | 答案 |
|------|------|
| 正交初始化有效吗? | ✅ 是的，比随机好6.0% |
| 冻结Q/K可行吗? | ❌ GPT上代价太大(-9.8%~-14.9%) |
| 目标达成了吗? | ❌ 所有数据集都超出5%目标 |
| 有速度优势吗? | ❌ 没有，与Baseline相同 |

### 9.2 失败原因分析

1. **任务类型差异**: 生成任务需要动态调整注意力
2. **上下文依赖**: GPT的Q/K需要适应变化的上下文
3. **预训练vs微调**: 冻结策略更适合微调而非预训练

### 9.3 下一步建议

1. **部分解冻**: 只解冻最后1-2层Q/K
2. **分层学习率**: 可训练参数使用更高学习率
3. **渐进解冻**: 训练过程中逐渐解冻Q/K
4. **仅限BERT**: 在论文中强调OELM仅适用于编码器任务

---

**报告生成时间**: 2026-02-12
**实验完成度**: 5/7 (71%)
