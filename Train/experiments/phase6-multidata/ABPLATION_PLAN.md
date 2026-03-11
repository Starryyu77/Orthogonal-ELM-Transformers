# Phase 6.5: 消融实验方案 (Ablation Study)

**实验阶段**: Phase 6.5 - 消融分析  
**基于**: Phase 5-6实验结果  
**目标**: 验证OELM各组件的独立贡献  
**创建时间**: 2026-03-11  
**执行地点**: NTU EEE GPU Cluster  

---

## 🎯 消融目标

### 核心问题
1. **正交初始化**是否必要？vs 随机初始化
2. **冻结Q/K**的独立贡献是多少？
3. **冻结FFN**的独立贡献是多少？
4. **学习率**对冻结策略的影响？

### 验证假设
- **H1**: 正交初始化显著优于随机初始化
- **H2**: 冻结Q/K是性能提升的主要来源
- **H3**: 冻结FFN在预训练后仍能保持性能
- **H4**: 冻结方法需要更高学习率

---

## 🔬 消融实验设计

### 实验组设计 (6组)

| 实验ID | 名称 | 冻结Q/K | 冻结FFN | 初始化 | 可训练参数 | 学习率 | 目的 |
|:------:|:-----|:-------:|:-------:|:------:|:----------:|:------:|:-----|
| A1 | **Baseline** | ❌ | ❌ | Xavier | 100% | 3e-4 | 基准 |
| A2 | **OELM-Full** | ✅ | ✅ | 正交 | 43.1% | 1e-3 | 完整方法 |
| A3 | **OELM-QK-Only** | ✅ | ❌ | 正交 | 88.6% | 1e-3 | 仅Q/K冻结 |
| A4 | **OELM-FFN-Only** | ❌ | ✅ | 正交 | 54.5% | 1e-3 | 仅FFN冻结 |
| A5 | **OELM-Random** | ✅ | ✅ | 随机 | 43.1% | 1e-3 | 正交必要性 |
| A6 | **OELM-LowLR** | ✅ | ✅ | 正交 | 43.1% | 3e-4 | 学习率影响 |

### 组件贡献分解公式

```
总效果 = A2 - A1
正交贡献 = A2 - A5
Q/K冻结贡献 = A3 - A1
FFN冻结贡献 = A4 - A1
交互效应 = A2 - (A3 + A4 - A1)
学习率影响 = A6 - A2
```

---

## 📊 详细配置

### A1: Baseline
```yaml
freeze_qk: false
freeze_ffn: false
qk_init_method: xavier
ffn_init_method: xavier
learning_rate: 3e-4
trainable_params: 124.4M (100%)
```

### A2: OELM-Full (完整方法)
```yaml
freeze_qk: true
freeze_ffn: true
qk_init_method: orthogonal
ffn_init_method: orthogonal
learning_rate: 1e-3
trainable_params: 53.6M (43.1%)
```

### A3: OELM-QK-Only
```yaml
freeze_qk: true
freeze_ffn: false
qk_init_method: orthogonal
ffn_init_method: xavier
learning_rate: 1e-3
trainable_params: 110.2M (88.6%)
```

### A4: OELM-FFN-Only
```yaml
freeze_qk: false
freeze_ffn: true
qk_init_method: xavier
ffn_init_method: orthogonal
learning_rate: 1e-3
trainable_params: 67.8M (54.5%)
```

### A5: OELM-Random (消融正交)
```yaml
freeze_qk: true
freeze_ffn: true
qk_init_method: normal  # 随机初始化
ffn_init_method: normal
learning_rate: 1e-3
trainable_params: 53.6M (43.1%)
```

### A6: OELM-LowLR (消融学习率)
```yaml
freeze_qk: true
freeze_ffn: true
qk_init_method: orthogonal
ffn_init_method: orthogonal
learning_rate: 3e-4  # 低学习率
trainable_params: 53.6M (43.1%)
```

---

## 📏 评估指标

### 主要指标
| 指标 | 说明 | 目标 |
|:-----|:-----|:-----|
| **Accuracy** | 验证集准确率 | 主要对比指标 |
| **Final Loss** | 验证集损失 | 辅助指标 |
| **Convergence Epoch** | 达到最佳性能所需epoch | 效率指标 |

### 消融分析指标
| 指标 | 计算方法 |
|:-----|:---------|
| **正交提升** | (A2 - A5) / A5 × 100% |
| **Q/K贡献度** | (A3 - A1) / (A2 - A1) × 100% |
| **FFN贡献度** | (A4 - A1) / (A2 - A1) × 100% |
| **学习率敏感度** | (A2 - A6) / A2 × 100% |

---

## 🚀 执行计划

### 实验数据集选择

选择**1个代表性数据集**进行完整消融，推荐**IMDB**或**AG News**：
- ✅ 已完成IMDB (Phase 5结果可用)
- 🆕 推荐AG News (多分类，与IMDB互补)

### 时间安排

| 阶段 | 时间 | 任务 | 实验组 |
|:-----|:-----|:-----|:-------|
| **Phase 1** | Day 1 | 修改代码支持随机初始化 | - |
| **Phase 2** | Day 2-3 | 运行全部6组实验 | A1-A6 |
| **Phase 3** | Day 4 | 数据分析与可视化 | - |
| **Phase 4** | Day 5 | 报告撰写 | - |

### 资源需求

| 资源 | 需求 | 说明 |
|:-----|:-----|:-----|
| GPU | 2× RTX A5000 | 并行运行 |
| 显存 | 24GB per GPU | 足够 |
| 存储 | ~50GB | 6组检查点 |
| 时间 | ~5天 | 包含分析 |

---

## 📈 预期结果与分析

### 预期性能排序

```
预期: A2 ≥ A3 > A1 > A4 > A5, A6

理想情况:
A2 (OELM-Full):     84.78% (参考Phase 5)
A3 (QK-Only):       84.06% (参考Phase 5)
A1 (Baseline):      82.95% (参考Phase 5)
A4 (FFN-Only):      ~82% (轻微下降)
A5 (Random):        ~81% (明显下降)
A6 (LowLR):         ~83% (学习率敏感)
```

### 组件贡献分解预期

| 组件 | 预期贡献 | 说明 |
|:-----|:---------|:-----|
| **正交初始化** | +3~5% | vs 随机初始化 |
| **Q/K冻结** | +1~2% | 主要贡献 |
| **FFN冻结** | -1~2% | 轻微损失 |
| **组合效应** | +2~3% | 协同作用 |

### 可视化计划

1. **柱状图**: 6组实验准确率对比
2. **饼图**: 各组件贡献度分解
3. **折线图**: 训练曲线对比
4. **热力图**: 不同配置组合效果

---

## 📁 输出目录结构

```
outputs/phase65-ablation/
├── experiments/
│   ├── A1_baseline/
│   │   ├── results.json
│   │   ├── best_model.pt
│   │   └── training.log
│   ├── A2_oelm_full/
│   ├── A3_oelm_qk_only/
│   ├── A4_oelm_ffn_only/
│   ├── A5_oelm_random/
│   └── A6_oelm_lowlr/
├── analysis/
│   ├── component_contribution.json
│   ├── ablation_summary.md
│   └── figures/
│       ├── accuracy_comparison.png
│       ├── component_breakdown.png
│       └── training_curves.png
└── report/
    └── ABLATION_REPORT.md
```

---

## 📝 代码修改清单

### 1. 模型代码修改

文件: `modeling_oelm_pretrain.py`

```python
# 添加随机初始化支持
class OELMConfig:
    qk_init_method: str = "orthogonal"  # "orthogonal" | "normal" | "xavier"
    ffn_init_method: str = "orthogonal"  # "orthogonal" | "normal" | "xavier"
    freeze_qk: bool = True
    freeze_ffn: bool = True

def initialize_weights(self):
    if self.config.qk_init_method == "orthogonal":
        self._orthogonal_init_qk()
    elif self.config.qk_init_method == "normal":
        self._normal_init_qk()
    elif self.config.qk_init_method == "xavier":
        self._xavier_init_qk()
    
    # 类似处理FFN...
```

### 2. 脚本修改

文件: `run_finetune.sh`

```bash
# 添加初始化方法参数
python finetune_from_pretrain.py \
    --qk_init_method ${QK_INIT:-"orthogonal"} \
    --ffn_init_method ${FFN_INIT:-"orthogonal"} \
    --freeze_qk ${FREEZE_QK:-"true"} \
    --freeze_ffn ${FREEZE_FFN:-"true"} \
    --learning_rate ${LR:-"1e-3"}
```

### 3. 批量提交脚本

文件: `run_ablation_study.sh`

```bash
#!/bin/bash
# 一键提交所有消融实验

# A1: Baseline
FREEZE_QK=false FREEZE_FFN=false LR=3e-4 \
    sbatch run_finetune.sh baseline

# A2: OELM-Full
FREEZE_QK=true FREEZE_FFN=true LR=1e-3 \
    sbatch run_finetune.sh oelm_full

# ... 其他组
```

---

## ✅ 成功标准

### 消融实验成功标准

| 条件 | 标准 | 置信度 |
|:-----|:-----|:------:|
| **正交必要性** | A2 > A5 (p < 0.05) | 必须有 |
| **Q/K有效性** | A3 > A1 | 必须有 |
| **学习率敏感** | A2 > A6 | 推荐有 |
| **可解释性** | 贡献度分解清晰 | 推荐有 |

### 论文级标准

- [ ] 6组实验全部完成
- [ ] 正交必要性统计显著
- [ ] 组件贡献度量化
- [ ] 训练曲线可视化
- [ ] 与Phase 5结果一致

---

## 🔗 相关文档

- [多数据集验证方案](./EXPERIMENT_PLAN.md)
- [Phase 5报告](../phase5-pretrain/FINAL_EXPERIMENT_REPORT.md)
- [消融实验脚本](../scripts/run_ablation_study.sh)

---

**计划制定**: 2026-03-11  
**最后更新**: 2026-03-11  
**执行状态**: 🟡 计划中
