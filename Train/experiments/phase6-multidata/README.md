# Phase 6-8: 验证与扩展实验完整方案

**文档索引** | **最后更新**: 2026-03-11  
**状态**: 🟡 方案已制定，等待执行

---

## 📚 文档导航

| 文档 | 路径 | 内容概要 | 优先级 |
|:-----|:-----|:---------|:------:|
| **实验总方案** | [EXPERIMENT_PLAN.md](./EXPERIMENT_PLAN.md) | Phase 6多数据集验证完整方案 | ⭐⭐⭐ |
| **消融实验** | [ABPLATION_PLAN.md](./ABPLATION_PLAN.md) | Phase 6.5组件消融设计 | ⭐⭐⭐ |
| **长期路线图** | [ROADMAP.md](./ROADMAP.md) | Phase 7-9扩展规划 | ⭐⭐ |
| **本索引** | [README.md](./README.md) | 文档导航与快速开始 | ⭐ |

---

## 🎯 快速开始

### 立即执行 (本周)

```bash
# 1. 登录集群
ssh tianyu016@10.97.216.128
cd /projects/LlamaFactory/OELM-Pretrain

# 2. AG News验证 (3种方法)
sbatch scripts/run_finetune_agnews_baseline.sh
sbatch scripts/run_finetune_agnews_qk.sh
sbatch scripts/run_finetune_agnews_qk_ffn.sh

# 3. SST-2验证
sbatch scripts/run_finetune_sst2_baseline.sh
sbatch scripts/run_finetune_sst2_qk.sh
sbatch scripts/run_finetune_sst2_qk_ffn.sh
```

### 关键实验配置

| 方法 | 学习率 | 可训练参数 | 冻结设置 |
|:-----|:------:|:----------:|:---------|
| Baseline | 3e-4 | 100% | 无 |
| OELM-QK | 1e-3 | 88.6% | Q/K |
| OELM-QK-FFN | 1e-3 | 43.1% | Q/K + FFN |

---

## 📊 实验矩阵

### Phase 6: 多数据集验证 (2周)

| 数据集 | 类别数 | 样本量 | 优先级 | 预计时间 |
|:-------|:------:|:------:|:------:|:--------:|
| AG News | 4 | 120K | P0 | 3天 |
| SST-2 | 2 | 67K | P0 | 2天 |
| XNLI | 3 | 392K | P1 | 4天 |
| MNLI | 3 | 392K | P1 | 4天 |

### Phase 6.5: 消融实验 (1周)

| 实验组 | 描述 | 关键对比 |
|:-------|:-----|:---------|
| A1 | Baseline | 基准 |
| A2 | OELM-Full | 完整方法 |
| A3 | OELM-QK-Only | 仅Q/K冻结 |
| A4 | OELM-FFN-Only | 仅FFN冻结 |
| A5 | OELM-Random | 消融正交 |
| A6 | OELM-LowLR | 消融学习率 |

### Phase 7-9: 扩展规划 (3个月)

| 阶段 | 内容 | 时间 | 里程碑 |
|:-----|:-----|:-----|:-------|
| Phase 7 | 模型规模扩展 (350M, 774M) | 5周 | M4: GPT-Medium验证 |
| Phase 8 | 生产级优化 | 3个月 | M7: 工程化完成 |
| Phase 9 | 论文撰写与投稿 | 并行 | M8: 论文投稿 |

---

## 🏆 成功标准

### 实验级标准

| 条件 | 标准 | 当前状态 |
|:-----|:-----|:---------|
| 多数据集验证 | ≥4个数据集 | 🟡 计划中 |
| 消融验证 | 正交必要性显著 | 🟡 计划中 |
| 参数效率 | ≤50%可训练参数 | ✅ Phase 5达成 |
| 性能保持 | ≥Baseline-1% | ✅ Phase 5达成 |

### 论文级标准

| 条件 | 标准 | 当前状态 |
|:-----|:-----|:---------|
| 数据集多样性 | 简单+复杂任务 | 🟡 Phase 6覆盖 |
| 规模验证 | 至少2个模型规模 | 🔵 Phase 7计划 |
| 消融完整性 | 组件贡献量化 | 🟡 Phase 6.5计划 |
| 统计显著性 | p<0.05 | 🟡 Phase 6包含 |
| 开源可复现 | GitHub完整 | ✅ 已发布 |

---

## 📈 预期成果

### 短期 (2周)
- [ ] AG News验证结果
- [ ] SST-2验证结果
- [ ] 消融实验结果

### 中期 (2个月)
- [ ] 4-6个数据集完整结果
- [ ] 350M模型验证
- [ ] 论文初稿

### 长期 (6个月)
- [ ] 774M大模型验证
- [ ] 生产级优化
- [ ] 顶会论文投稿

---

## 🔗 相关链接

### 历史实验

| 实验 | 路径 | 关键结果 |
|:-----|:-----|:---------|
| Phase 5 | [FINAL_EXPERIMENT_REPORT.md](../phase5-pretrain/FINAL_EXPERIMENT_REPORT.md) | OELM-QK-FFN: 84.78% (+1.83%) |
| Phase 4 | [REPORT.md](../phase4-gpt-classification/REPORT.md) | GPT分类: +8.14%平均 |
| Phase 3 | [REPORT.md](../phase3-gpt-ablation/REPORT.md) | GPT生成: 不适用 |

### 代码仓库

- **GitHub**: https://github.com/Starryyu77/Orthogonal-ELM-Transformers
- **本地路径**: `/projects/LlamaFactory/OELM-Pretrain`

---

## 📞 联系信息

| 角色 | 职责 | 联系方式 |
|:-----|:-----|:---------|
| 主要研究员 | 实验执行、分析 | tianyu016 |
| GPU集群 | 计算资源 | NTU EEE GPU Cluster |

---

**文档创建**: 2026-03-11  
**文档版本**: v1.0  
**下次更新**: Phase 6完成后
