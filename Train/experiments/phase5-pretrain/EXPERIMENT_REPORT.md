# OELM Pretrain 实验报告

**实验日期**: 2026-03-09 ~ 2026-03-11  
**实验地点**: NTU EEE GPU Cluster  
**实验人员**: tianyu016  

---

## 1. 实验目标

验证 **OELM (Orthogonal ELM)** 方法在**预训练+微调**完整流程中的有效性。

### 核心假设
- **预训练阶段**: Q/K冻结+正交初始化是否能提供良好的初始化？
- **下游任务**: 更少参数的预训练模型能否达到与全参数Baseline相当的效果？

---

## 2. 实验设计

### 2.1 预训练阶段

| 方法 | 冻结Q/K | 冻结FFN | 可训练参数 | 学习率 | 数据集 | Epochs |
|------|---------|---------|-----------|--------|--------|--------|
| **Baseline** | ❌ | ❌ | 124.4M (100%) | 3e-4 | TinyStories | 1 |
| **OELM-QK** | ✅ | ❌ | 110.2M (88.6%) | 1e-3 | TinyStories | 1 |
| **OELM-QK-FFN** | ✅ | ✅ | 53.6M (43.1%) | 1e-3 | TinyStories | 1 |

**模型配置**: GPT-Small (d_model=768, layers=12, heads=12)

### 2.2 下游微调阶段

| 方法 | 数据集 | Batch Size | Epochs | 学习率 | 冻结Q/K |
|------|--------|-----------|--------|--------|---------|
| **Baseline** | IMDB | 16 | 3 | 3e-4 | ❌ |
| **OELM-QK** | IMDB | 16 | 3 | 1e-3 | ✅ |
| **OELM-QK-FFN** | IMDB | 16 | 3 | 1e-3 | ✅ |

---

## 3. 实验过程

### 3.1 遇到的问题与解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| GPU配额限制 | 申请2块Pro 6000 | 改为申请1块 |
| ImportError | CosineLRScheduler不存在 | 改为LambdaLR |
| DataLoader错误 | collate_fn resize问题 | 修复为预分配tensor |
| 磁盘空间不足 | 检查点过多 (>100个) | 清理中间检查点，释放~200GB |
| 训练超步数 | 恢复训练epoch计算问题 | 监控并及时停止 |

### 3.2 关键时间节点

| 时间 | 事件 |
|------|------|
| 03-09 08:23 | Baseline预训练启动 |
| 03-09 21:42 | Baseline预训练完成 (7h34m) |
| 03-10 08:23 | OELM-QK和OELM-QK-FFN启动 |
| 03-10 15:27 | OELM训练失败（磁盘满） |
| 03-10 16:35 | 清理磁盘，恢复训练 |
| 03-10 22:08 | OELM训练再次失败（超步数） |
| 03-11 02:06 | 最终清理，准备微调 |

---

## 4. 预训练结果

### 4.1 训练指标

| 方法 | 总步数 | Loss | PPL | 训练时间 |
|------|--------|------|-----|----------|
| **Baseline** | 132,483 | 0.8592 | 2.36 | 7h34m |
| **OELM-QK** | ~196,000* | - | - | ~10h |
| **OELM-QK-FFN** | ~216,000* | - | - | ~10h |

*注: 由于恢复训练机制问题，实际训练步数超过目标

### 4.2 模型文件

| 方法 | Best模型路径 | 大小 |
|------|-------------|------|
| Baseline | `outputs/pretrain/baseline/best/pytorch_model.pt` | 475M |
| OELM-QK | `outputs/pretrain/oelm_qk/best/pytorch_model.pt` | 475M |
| OELM-QK-FFN | `outputs/pretrain/oelm_qk_ffn/best/pytorch_model.pt` | 475M |

---

## 5. 下游微调（进行中）

### 5.1 微调作业

| 作业ID | 方法 | 状态 | 启动时间 |
|--------|------|------|----------|
| 44374 | Baseline | Running | 03-11 02:06 |
| 44375 | OELM-QK | Running | 03-11 02:06 |
| 44376 | OELM-QK-FFN | Pending | 03-11 02:06 |

### 5.2 预期结果

| 方法 | 可训练参数 | 预期准确率 | 参数效率 |
|------|-----------|-----------|----------|
| Baseline | 100% | 基准 | 1.0x |
| OELM-QK | 88.6% | ≥ 基准-2% | 1.13x |
| OELM-QK-FFN | 43.1% | ≥ 基准-2% | 2.32x |

---

## 6. 关键发现

### 6.1 预训练阶段

1. **OELM-QK-FFN Loss更低**: 在恢复训练过程中，OELM-QK-FFN的Loss (0.35) 比OELM-QK (0.82) 更低
2. **参数效率显著**: OELM-QK-FFN仅用43.1%参数，但训练稳定
3. **磁盘管理重要**: 大规模预训练需要及时清理检查点

### 6.2 技术经验

1. **恢复训练需谨慎**: 恢复时epoch计算可能重复，需要监控
2. **磁盘配额限制**: 300GB配额对大规模实验是瓶颈
3. **检查点策略**: 保存间隔2000步合适，但需要定期清理

---

## 7. 下一步工作

1. **等待微调完成** (~2小时)
2. **收集微调结果**: 准确率、F1分数
3. **对比分析**: Baseline vs OELM-QK vs OELM-QK-FFN
4. **扩展到其他数据集**: AG News, XNLI, MNLI

---

## 8. 实验结论（待完成微调后更新）

**待验证假设**:
- OELM-QK能否在88.6%参数下达到Baseline性能？
- OELM-QK-FFN能否在43.1%参数下达到Baseline性能？

**预期结论**: 
如果OELM-QK-FFN能在43.1%参数下达到与Baseline相当的准确率，则证明OELM方法在预训练阶段同样有效。

---

## 附录

### 关键文件位置

```
/projects/LlamaFactory/OELM-Pretrain/
├── models/modeling_oelm_pretrain.py
├── scripts/
│   ├── train_pretrain.py
│   ├── finetune_from_pretrain.py
│   └── run_*.sh
├── outputs/pretrain/
│   ├── baseline/best/pytorch_model.pt
│   ├── oelm_qk/best/pytorch_model.pt
│   └── oelm_qk_ffn/best/pytorch_model.pt
└── outputs/finetune/
    └── (进行中)
```

### 监控命令

```bash
# 查看作业状态
ssh ntu016@10.97.216.128
watch squeue -u tianyu016

# 查看微调进度
tail -f /projects/LlamaFactory/OELM-Pretrain/logs/finetune_*-*.out
```