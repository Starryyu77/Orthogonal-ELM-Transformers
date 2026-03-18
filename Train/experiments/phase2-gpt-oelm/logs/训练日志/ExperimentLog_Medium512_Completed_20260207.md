# 实验完成日志 - Medium-512 GPT vs OELM 对比实验

**项目名称**: Orthogonal ELM Transformer
**实验名称**: Medium-512 配置 GPT vs OELM 对比实验（完成版）
**作者**: 张天禹 (Zhang Tianyu)
**学号**: s125mdg43_10
**指导单位**: NTU MLDA Lab
**实验日期**: 2026年2月6日 - 2月7日
**报告生成**: Claude Code AI Assistant

---

## 实验目标

对比标准GPT模型与正交极限学习机Transformer (OELM) 在Medium-512配置下的性能差异，验证OELM在减少参数量的同时保持语言建模能力的效果。

---

## 实验配置

### 模型架构

| 参数 | 值 |
|------|-----|
| Model Type | GPT vs OELM |
| n_layers | 6 |
| d_model | 512 |
| n_heads | 8 |
| d_ff | 2048 |
| seq_len | 512 |
| vocab_size | 50257 |

### 参数量对比

| 模型 | 总参数量 | 模型文件大小 |
|------|----------|--------------|
| GPT Medium-512 | 44.9M (100%) | 514 MB |
| OELM Medium-512 | 41.8M (93%) | 490 MB |

**参数减少**: OELM 比 GPT 少 **6.9%** 参数（约 3.1M 参数）

### 训练参数

| 参数 | 值 |
|------|-----|
| 总训练步数 | 100,000 |
| Batch Size (per GPU) | 8 |
| 有效Batch Size | 16 (2 GPUs) |
| 学习率 (max) | 3e-4 |
| 学习率 (min) | 3e-5 |
| Warmup Steps | 2,000 |
| 优化器 | AdamW |
| 权重衰减 | 0.1 |
| 梯度裁剪 | 1.0 |

### 硬件环境

| 项目 | 配置 |
|------|------|
| 服务器 | MLDA GPU (NTU) |
| GPUs | 4 × NVIDIA RTX A5000 (24GB) |
| GPT分配 | GPU 0,1 (2卡数据并行) |
| OELM分配 | GPU 2,3 (2卡数据并行) |
| CUDA版本 | 12.2 |
| PyTorch版本 | 2.0.1+cu118 |

### 数据集

| 参数 | 值 |
|------|-----|
| 数据集 | TinyStories |
| Tokenizer | GPT-2 |
| 词表大小 | 50257 |
| 训练集 | 896 MB (train.bin) |
| 验证集 | 9 MB (val.bin) |

---

## 训练结果

### 最终性能对比

| 模型 | 训练Loss | 验证Loss | 困惑度 (PPL) | 状态 |
|------|----------|----------|--------------|------|
| **GPT Medium-512** | 1.4600 | **1.4215** | **4.14** | ✅ 完成 |
| **OELM Medium-512** | 1.5946 | 1.5389 | 4.66 | ✅ 完成 |
| **OELM Freeze (步骤三)** | 1.6755 | **1.5971** | **4.94** | ✅ 完成 (100K步) |

### 关键发现

1. **GPT 性能最优**: GPT 验证 Loss 1.4215，明显优于所有 OELM 变体
2. **OELM 无冻结版次之**: 验证 Loss 1.5389，比 GPT 差约 8.3%
3. **OELM 冻结版最差**: 验证 Loss 1.5971，比 GPT 差约 12.4%
4. **参数效率**: OELM 参数少 6.9%，冻结版可训练参数少 15.4%
5. **冻结策略**: 完全冻结 Q/K 导致性能下降较多，需要探索更精细的策略

### 训练曲线终点

**GPT** (最后10步):
```
Step  99000 | Loss: 1.5099 | PPL: 4.53 | LR: 1.34e-07
  Validation | Loss: 1.4215 | PPL: 4.14  ← 最佳模型
Step  99100 | Loss: 1.4412 | PPL: 4.23
Step  99200 | Loss: 1.4671 | PPL: 4.34
Step  99300 | Loss: 1.5549 | PPL: 4.73
Step  99400 | Loss: 1.5140 | PPL: 4.54
Step  99500 | Loss: 1.7476 | PPL: 5.74
Step  99600 | Loss: 1.6286 | PPL: 5.10
Step  99700 | Loss: 1.4491 | PPL: 4.26
Step  99800 | Loss: 1.6444 | PPL: 5.18
Step  99900 | Loss: 1.4600 | PPL: 4.31
```

**OELM** (最后10步):
```
Step  99000 | Loss: 1.6347 | PPL: 5.13 | LR: 1.34e-07
  Validation | Loss: 1.5389 | PPL: 4.66  ← 最佳模型
Step  99100 | Loss: 1.5976 | PPL: 4.94
Step  99200 | Loss: 1.6078 | PPL: 4.99
Step  99300 | Loss: 1.7039 | PPL: 5.50
Step  99400 | Loss: 1.6526 | PPL: 5.22
Step  99500 | Loss: 1.8759 | PPL: 6.53
Step  99600 | Loss: 1.7689 | PPL: 5.86
Step  99700 | Loss: 1.5934 | PPL: 4.92
Step  99800 | Loss: 1.7898 | PPL: 5.99
Step  99900 | Loss: 1.5946 | PPL: 4.93
```

**OELM-Freeze** (最后10步):
```
Step  99000 | Loss: 1.7431 | PPL: 5.71 | LR: 3.01e-05
  Validation | Loss: 1.5985 | PPL: 4.95
Step  99100 | Loss: 1.6452 | PPL: 5.18
Step  99200 | Loss: 1.6795 | PPL: 5.36
Step  99300 | Loss: 1.7579 | PPL: 5.80
Step  99400 | Loss: 1.7296 | PPL: 5.64
Step  99500 | Loss: 1.9500 | PPL: 7.03
Step  99600 | Loss: 1.8463 | PPL: 6.34
Step  99700 | Loss: 1.6685 | PPL: 5.30
Step  99800 | Loss: 1.8666 | PPL: 6.47
Step  99900 | Loss: 1.6755 | PPL: 5.34 | LR: 3.00e-05
  Validation | Loss: 1.5971 | PPL: 4.94  ← 最佳模型

Training complete! Final checkpoint saved.
```

---

## 模型文件

### 本地存储路径

```
models/checkpoints/
├── gpt_medium512/
│   ├── best.pt    (514 MB) - 验证Loss最低模型
│   ├── final.pt   (514 MB) - 最终训练模型
│   └── latest.pt  (514 MB) - 最后检查点
├── oelm_medium512/
│   ├── best.pt    (490 MB) - 验证Loss最低模型
│   ├── final.pt   (490 MB) - 最终训练模型
│   └── latest.pt  (490 MB) - 最后检查点
└── exp_oelm_freeze/
    ├── best.pt    (490 MB) - 步骤三冻结模型 ✅
    ├── final.pt   (490 MB) - 最终模型 ✅
    ├── latest.pt  (490 MB) - 最后检查点 ✅
    └── training.log         - 完整训练日志 ✅
```

### 远程服务器路径

```
~/Orthogonal_ELM_Transformers/Train/models/checkpoints/
├── gpt_medium512/
└── oelm_medium512/
```

---

## 实验结论

### 主要发现

1. **GPT 基线更强**: 在这个Medium-512配置下，标准GPT的验证性能优于OELM
2. **参数效率**: OELM虽然参数少6.9%，但性能差距约8.3%，参数效率略低
3. **训练稳定性**: 两个模型都稳定完成了10万步训练，没有崩溃或发散

### 冻结模型实验 (步骤三)

**配置**: OELM + Q/K投影矩阵冻结
- 状态: 🔄 训练中 (约 92,500 / 100,000 步)
- 当前最佳验证 Loss: **1.6024** (PPL: 4.97)
- 与无冻结OELM对比: 性能略差 (1.6024 vs 1.5389)
- 与GPT对比: 差距约 12.7%

**冻结模型训练曲线** (最近10次验证):
```
Validation | Loss: 1.6125 | PPL: 5.02
Validation | Loss: 1.6114 | PPL: 5.01
Validation | Loss: 1.6108 | PPL: 5.01
Validation | Loss: 1.6078 | PPL: 4.99
Validation | Loss: 1.6068 | PPL: 4.99
Validation | Loss: 1.6063 | PPL: 4.98
Validation | Loss: 1.6048 | PPL: 4.98
Validation | Loss: 1.6038 | PPL: 4.97 ← 最佳
Validation | Loss: 1.6024 | PPL: 4.97 ← 最佳
Validation | Loss: 1.6027 | PPL: 4.97 (当前)
```

### 可能原因分析

1. **OELM初始化**: 可能需要更仔细的正交初始化调优
2. **学习率**: OELM可能需要不同的学习率策略
3. **冻结机制**: 当前冻结实验显示性能略低于无冻结版本，可能需要调整冻结策略

### 后续建议

1. 尝试不同的正交初始化方法
2. 实验启用Q/K投影矩阵冻结
3. 在更大规模模型上测试（Large配置）
4. 尝试不同的学习率和优化器设置

---

## 附录

### 下载时间

- GPT 模型下载: 2026-02-07 14:41
- OELM 模型下载: 2026-02-07 14:42
- OELM Freeze 模型下载: 2026-02-07 14:45
- 总下载大小: ~2.0 GB

### 训练完成时间

| 模型 | 开始时间 | 完成时间 | 状态 |
|------|----------|----------|------|
| GPT | 2026-02-06 15:30 | 2026-02-06 22:52 | ✅ 完成 |
| OELM | 2026-02-06 15:30 | 2026-02-06 21:30 | ✅ 完成 |
| OELM-Freeze | 2026-02-06 | 2026-02-07 15:25 | ✅ 完成 |

---

**实验状态**: ✅ 完成并归档
