# Phase 6: 多数据集验证实验方案

**实验阶段**: Phase 6 - 扩展验证  
**基于**: Phase 5 OELM预训练模型  
**目标**: 在多个下游任务上验证OELM-QK-FFN的泛化能力  
**创建时间**: 2026-03-11  
**执行地点**: NTU EEE GPU Cluster  

---

## 🎯 实验目标

### 主要目标
1. 验证预训练OELM模型在多个下游任务上的有效性
2. 确认OELM-QK-FFN方法具有跨任务泛化能力
3. 建立多数据集基准，支持论文结论

### 验证假设
- **H1**: OELM-QK-FFN在多个分类任务上超越或持平Baseline
- **H2**: 参数效率(43.1%可训练参数)在不同任务上保持稳定
- **H3**: 正交初始化对性能提升有显著贡献

---

## 📊 验证数据集与评估指标

### 数据集选择矩阵

| 优先级 | 数据集 | 任务类型 | 类别数 | 训练集 | 验证集 | 领域 | 难度 |
|:------:|:-------|:---------|:------:|:------:|:------:|:-----|:----:|
| **P0** | AG News | 新闻分类 | 4 | 120K | 7.6K | 新闻 | ⭐⭐ |
| **P0** | SST-2 | 情感分析 | 2 | 67K | 872 | 电影评论 | ⭐ |
| **P1** | XNLI | 自然语言推理 | 3 | 392K | 2.5K | 通用 | ⭐⭐⭐ |
| **P1** | MNLI | 自然语言推理 | 3 | 392K | 9.8K | 通用 | ⭐⭐⭐ |
| **P2** | Yelp Review | 情感分析 | 2 | 560K | 38K | 商业评论 | ⭐⭐ |
| **P2** | DBpedia | 实体分类 | 14 | 560K | 70K | 百科知识 | ⭐⭐⭐ |

### 选择理由

**P0 - 必须完成**:
- **AG News**: 多分类(4类)，与IMDB(2类)形成互补，验证多分类能力
- **SST-2**: 短文本情感分析，与IMDB长文本形成对比，验证不同文本长度

**P1 - 推荐完成**:
- **XNLI**: 自然语言推理，复杂语义理解，Phase 4已验证OELM在此类任务上效果显著(+11.60%)
- **MNLI**: 更大规模NLI，与XNLI形成验证集大小对比

**P2 - 扩展验证**:
- **Yelp**: 大规模情感分析，与IMDB/SST-2形成领域三剑客
- **DBpedia**: 细粒度分类(14类)，验证复杂分类能力

---

## 📏 评估指标体系

### 主要指标

| 指标 | 计算公式 | 说明 | 优先级 |
|:-----|:---------|:-----|:------:|
| **Accuracy** | correct / total | 整体准确率 | ⭐⭐⭐ |
| **Macro-F1** | mean(F1 per class) | 类别不平衡稳健 | ⭐⭐⭐ |
| **Weighted-F1** | Σ(support_i × F1_i) / total | 考虑类别分布 | ⭐⭐ |
| **Per-class F1** | F1 for each class | 细粒度分析 | ⭐ |

### 效率指标

| 指标 | 说明 | 计算方式 | 优先级 |
|:-----|:-----|:---------|:------:|
| **Trainable Params** | 可训练参数比例 | 可训练 / 总参数 | ⭐⭐⭐ |
| **Training Time** | 训练时间 | wall clock time | ⭐⭐ |
| **Throughput** | 样本/秒 | samples/sec | ⭐ |
| **Convergence Speed** | 达到90%最佳性能所需epoch | early stopping | ⭐⭐ |

### 统计显著性

| 测试 | 方法 | 目的 |
|:-----|:-----|:-----|
| **配对t检验** | 重复运行3次 | 验证结果稳定性 |
| **置信区间** | 95% CI | 量化不确定性 |
| **效应量** | Cohen's d | 实际意义大小 |

---

## 🔬 实验配置

### 模型配置 (三种方法)

```yaml
# 方法1: Baseline
freeze_qk: false
freeze_ffn: false
init_method: normal
learning_rate: 3e-4
trainable_params: 100%

# 方法2: OELM-QK
freeze_qk: true
freeze_ffn: false
init_method: orthogonal
learning_rate: 1e-3
trainable_params: 88.6%

# 方法3: OELM-QK-FFN
freeze_qk: true
freeze_ffn: true
init_method: orthogonal
learning_rate: 1e-3
trainable_params: 43.1%
```

### 训练配置

| 配置项 | 值 | 说明 |
|:-------|:---|:-----|
| **预训练模型** | GPT-Small (124M) | Phase 5产出 |
| **d_model** | 768 | 与预训练一致 |
| **num_layers** | 12 | 与预训练一致 |
| **num_heads** | 12 | 与预训练一致 |
| **max_seq_len** | 512 | 标准分类长度 |
| **batch_size** | 16 | 显存优化 |
| **num_epochs** | 3 | 早期收敛 |
| **warmup_steps** | 500 | 稳定训练 |
| **weight_decay** | 0.01 | 正则化 |
| **pooling** | last token | 分类标准做法 |
| **optimizer** | AdamW | 标准 |
| **scheduler** | CosineAnnealing | 平滑衰减 |

### 数据集特定配置

| 数据集 | max_seq_len | batch_size | 特殊处理 |
|:-------|:-----------:|:----------:|:---------|
| AG News | 512 | 16 | 标准 |
| SST-2 | 128 | 32 | 短文本，增大batch |
| XNLI | 512 | 16 | premise + [SEP] + hypothesis |
| MNLI | 512 | 16 | premise + [SEP] + hypothesis |
| Yelp | 512 | 16 | 标准 |
| DBpedia | 256 | 32 | 标题+描述，较短 |

---

## 📈 结果对比矩阵

### 主实验结果表

| 数据集 | 类别数 | Baseline Acc | OELM-QK Acc | OELM-QK-FFN Acc | 最佳方法 | 相对提升 |
|:-------|:------:|:------------:|:-----------:|:---------------:|:--------:|:--------:|
| IMDB | 2 | 82.95% | 84.06% | **84.78%** | QK-FFN | +1.83% |
| AG News | 4 | - | - | - | - | - |
| SST-2 | 2 | - | - | - | - | - |
| XNLI | 3 | - | - | - | - | - |
| MNLI | 3 | - | - | - | - | - |
| **平均** | - | - | - | - | - | - |

### 参数效率表

| 方法 | 可训练参数 | 相对参数量 | 预期速度提升 |
|:-----|:----------:|:----------:|:------------:|
| Baseline | 124.4M | 100% | 1.0x |
| OELM-QK | 110.2M | 88.6% | ~1.1x |
| OELM-QK-FFN | 53.6M | 43.1% | ~1.5x |

---

## 🚀 执行计划

### Week 1: P0数据集

| 日期 | 任务 | 数据集 | 方法 | GPU |
|:-----|:-----|:-------|:-----|:----:|
| Day 1 | 准备脚本 | - | - | - |
| Day 2-3 | 运行实验 | AG News | 3种 | 2块 |
| Day 4-5 | 运行实验 | SST-2 | 3种 | 2块 |
| Day 6-7 | 分析结果 | - | - | - |

### Week 2: P1数据集

| 日期 | 任务 | 数据集 | 方法 | GPU |
|:-----|:-----|:-------|:-----|:----:|
| Day 8-10 | 运行实验 | XNLI | 3种 | 2块 |
| Day 11-13 | 运行实验 | MNLI | 3种 | 2块 |
| Day 14 | 中期总结 | - | - | - |

### Week 3: 消融实验 (Phase 6.5)

详见 [ABPLATION_PLAN.md](./ABPLATION_PLAN.md)

---

## 📁 输出目录结构

```
outputs/phase6-multidata/
├── ag_news/
│   ├── baseline/
│   │   ├── results.json
│   │   └── best_model.pt
│   ├── oelm_qk/
│   │   ├── results.json
│   │   └── best_model.pt
│   └── oelm_qk_ffn/
│       ├── results.json
│       └── best_model.pt
├── sst2/
│   ├── baseline/
│   ├── oelm_qk/
│   └── oelm_qk_ffn/
├── xnli/
│   └── ...
├── mnli/
│   └── ...
├── summary/
│   ├── all_results.json
│   ├── comparison_table.md
│   └── statistical_tests.json
└── logs/
    ├── ag_news_baseline.log
    ├── ag_news_oelm_qk.log
    └── ...
```

---

## ✅ 成功标准

### 实验成功标准

| 条件 | 标准 | 说明 |
|:-----|:-----|:-----|
| **主要成功** | OELM-QK-FFN ≥ Baseline - 1% | 在至少3个数据集上达成 |
| **完全成功** | OELM-QK-FFN > Baseline | 在至少2个数据集上达成 |
| **参数效率** | 参数量 ≤ 50% | OELM-QK-FFN必须达成 |
| **统计显著** | p < 0.05 | 配对t检验 |

### 论文级标准

- [ ] 至少4个数据集验证
- [ ] 包含简单和复杂任务
- [ ] 消融实验验证正交必要性
- [ ] 统计显著性检验
- [ ] 完整的超参数搜索记录

---

## 🔗 相关文档

- [Phase 5报告](../phase5-pretrain/FINAL_EXPERIMENT_REPORT.md)
- [消融实验方案](./ABPLATION_PLAN.md)
- [执行脚本](../scripts/)

---

**计划制定**: 2026-03-11  
**最后更新**: 2026-03-11  
**执行状态**: 🟡 计划中
