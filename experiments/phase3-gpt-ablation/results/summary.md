# Phase 3: 实验结果汇总

## 执行摘要

| 数据集 | Baseline PPL | OELM-Freeze PPL | 差距 | 目标达成? |
|--------|--------------|-----------------|------|-----------|
| TinyStories | 4.27 | 4.69 | **+9.8%** | ❌ |
| OpenWebText | 47.24 | 54.29 | **+14.9%** | ❌ |
| WikiText-103 | TBD | TBD | TBD | ⏳ |

**结论**: OELM-Freeze 在所有已完成的数据集上均未达到 ≤5% 的目标。

---

## 详细结果

### TinyStories 完整消融

| 实验 | 方法 | PPL | vs Baseline | 训练时间 | 状态 |
|------|------|-----|-------------|----------|------|
| GPT-01 | Baseline | 4.27 | 基准 | 4h 31m | ✅ |
| GPT-02 | OELM-Freeze | 4.69 | **+9.8%** | 4h 32m | ✅ |
| GPT-03 | OELM-Random | 4.97 | **+16.4%** | 4h 15m | ✅ |

**消融分析**:
```
Baseline → OELM-Freeze:  +9.8% (冻结Q/K + 正交init)
Baseline → OELM-Random:  +16.4% (冻结Q/K + 随机init)
OELM-Random → OELM-Freeze: -6.0% (正交init的价值)
```

**关键发现**:
- ✅ 正交初始化有效 (提升6.0%)
- ❌ 但冻结Q/K本身代价较大

---

### OpenWebText 对比

| 实验 | 方法 | PPL | vs Baseline | 训练时间 | 状态 |
|------|------|-----|-------------|----------|------|
| GPT-04 | Baseline | 47.24 | 基准 | 9h 10m | ✅ |
| GPT-05 | OELM-Freeze | 54.29 | **+14.9%** | 9h 8m | ✅ |

**关键发现**:
- 差距比 TinyStories 更大 (+14.9% vs +9.8%)
- 训练速度几乎相同 (无计算优势)

---

### 跨数据集趋势

| 数据集 | 规模 | Baseline | OELM-Freeze | 差距 | 趋势 |
|--------|------|----------|-------------|------|------|
| TinyStories | 小 (50M) | 4.27 | 4.69 | +9.8% | 小差距 |
| OpenWebText | 大 (40B) | 47.24 | 54.29 | +14.9% | 大差距 |

**推论**: 数据集越复杂，冻结Q/K的代价越大。

---

## 训练速度对比

| 实验 | 平均步时间 | 总训练时间 | 备注 |
|------|-----------|-----------|------|
| GPT-01 | 0.084s | 4h 31m | Baseline |
| GPT-02 | 0.084s | 4h 32m | OELM-Freeze |
| GPT-03 | 0.077s | 4h 15m | OELM-Random (略快) |
| GPT-04 | 0.184s | 9h 10m | Baseline |
| GPT-05 | 0.184s | 9h 8m | OELM-Freeze |

**结论**: OELM-Freeze 无训练速度优势。

---

## 原始数据文件

```
gpt-oelm-project/outputs/
├── GPT-01_baseline/
│   ├── timing_stats.json          # 完整计时统计
│   ├── training.log               # 训练日志
│   └── best.pt                    # 最佳模型 (514MB)
├── GPT-02_oelm_freeze/
│   ├── timing_stats.json
│   ├── training.log
│   └── best.pt                    # 最佳模型 (490MB)
├── GPT-03_oelm_random/
│   ├── timing_stats.json
│   ├── training.log
│   └── best.pt
├── GPT-04_openwebtext_baseline/
│   ├── timing_stats.json
│   ├── training.log
│   └── best.pt
└── GPT-05_openwebtext_oelm/
    ├── timing_stats.json
    ├── training.log
    └── best.pt
```

---

## 结论与建议

### 主要结论

1. **正交初始化有价值**: OELM-Freeze 比 OELM-Random 好 6.0%
2. **冻结Q/K代价大**: 即使正交init也无法完全弥补
3. **GPT比BERT敏感**: XNLI上+1.08%优势 → GPT上-9.8%~-14.9%劣势
4. **无速度优势**: 训练时间与Baseline相同

### 建议改进方向

1. **部分解冻**: 只冻结部分层的Q/K
2. **渐进式解冻**: 训练后期解冻Q/K
3. **分层学习率**: 不同层使用不同学习率
4. **调整学习率**: 当前1e-3可能不够优化

---

*最后更新: 2026-02-10*
