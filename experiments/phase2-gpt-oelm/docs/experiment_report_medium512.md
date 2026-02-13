# OELM Medium-512 对比实验报告

## 实验概述

本实验旨在对比 **标准GPT模型** 与 **正交极限学习机Transformer (OELM)** 在相同架构配置下的性能差异，验证OELM在减少参数量的同时保持语言建模能力的效果。

---

## 1. 模型架构

### 1.1 通用配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_layers | 6 | Transformer层数 |
| d_model | 512 | 模型维度/隐藏层大小 |
| n_heads | 8 | 注意力头数 |
| d_head | 64 | 每个头的维度 (d_model / n_heads) |
| d_ff | 2048 | 前馈网络维度 (4 × d_model) |
| seq_len | 512 | 序列长度 |
| vocab_size | 50257 | GPT-2词表大小 |

### 1.2 基准模型：标准GPT (GPT Medium-512)

```
GPT Model:
├── Embedding Layer: 50257 × 512 = 25,731,584 params
├── 6 × Transformer Block:
│   ├── Attention:
│   │   ├── Q_proj: 512 × 512 = 262,144
│   │   ├── K_proj: 512 × 512 = 262,144
│   │   ├── V_proj: 512 × 512 = 262,144
│   │   └── Output_proj: 512 × 512 = 262,144
│   └── FeedForward:
│       ├── Up_proj: 512 × 2048 = 1,048,576
│       └── Down_proj: 2048 × 512 = 1,048,576
└── LayerNorm & Biases

Total parameters: 44,896,768 (约44.9M)
Trainable parameters: 44,896,768 (100%)
```

### 1.3 实验模型：OELM (Orthogonal ELM Transformer)

OELM的核心创新在于**冻结Query和Key投影矩阵的正交初始化**：

```
OELM Model:
├── Embedding Layer: 50257 × 512 = 25,731,584 params (同GPT)
├── 6 × Orthogonal Transformer Block:
│   ├── Attention:
│   │   ├── Q_proj: 512 × 512 = 262,144 [FROZEN - 正交初始化]
│   │   ├── K_proj: 512 × 512 = 262,144 [FROZEN - 正交初始化]
│   │   ├── V_proj: 512 × 512 = 262,144 [TRAINABLE]
│   │   └── Output_proj: 512 × 512 = 262,144 [TRAINABLE]
│   └── FeedForward:
│       ├── Up_proj: 512 × 2048 = 1,048,576 [TRAINABLE]
│       └── Down_proj: 2048 × 512 = 1,048,576 [TRAINABLE]
└── LayerNorm & Biases

Total parameters: 41,751,040 (约41.8M)
Trainable parameters: 41,751,040 (100% - 当前实现)
Frozen parameters: 0 (当前实现中未启用冻结)
```

### 1.4 参数量对比

| 模型 | 总参数量 | 相比GPT | 节省比例 |
|------|----------|---------|----------|
| GPT Medium-512 | 44.9M | 100% | 0% |
| OELM Medium-512 | 41.8M | 93.0% | **7.0%** |

**注意**：当前OELM实现中冻结参数功能未启用，理论上冻结Q/K投影矩阵可减少约50%的注意力层参数量。

---

## 2. 实验方法

### 2.1 数据集

- **数据集**: TinyStories
- **数据规模**: 约 2.3M 条短篇故事
- **Tokenized大小**: Train 896MB, Val 9MB
- **Tokenizer**: GPT-2 (vocab_size=50257)

### 2.2 训练配置

| 配置项 | 值 |
|--------|-----|
| 训练步数 | 100,000 |
| Batch Size (per GPU) | 8 |
| 有效Batch Size | 16 (2 GPU × 8) |
| 学习率 (max) | 3e-4 |
| 学习率 (min) | 3e-5 |
| Warmup Steps | 2,000 |
| 优化器 | AdamW |
| 权重衰减 | 0.1 |
| 梯度裁剪 | 1.0 |
| 验证间隔 | 1,000 steps |
| 保存间隔 | 5,000 steps |

### 2.3 硬件环境

| 项目 | 配置 |
|------|------|
| 服务器 | MLDA GPU (NTU) |
| GPU | 4 × NVIDIA RTX A5000 (24GB) |
| CUDA版本 | 12.2 |
| PyTorch版本 | 2.0.1+cu118 |

### 2.4 分布式训练设置

- **GPT**: GPU 0,1 (2卡数据并行)
- **OELM**: GPU 2,3 (2卡数据并行)
- **启动命令**: `torch.distributed.run --nproc_per_node=2`

---

## 3. 实验结果

### 3.1 训练进度 (截至Step 3800)

| Step | GPT Loss | GPT PPL | OELM Loss | OELM PPL |
|------|----------|---------|-----------|----------|
| 0 | 10.93 | 22026 | 10.91 | 22026 |
| 1000 | 3.75 | 42.55 | 3.85 | 46.90 |
| 2000 | 3.05 | 21.09 | - | - |
| 3000 | 2.54 | 12.68 | 2.80 | 16.37 |
| 3800 | - | - | 2.50 | 12.13 |

### 3.2 验证集性能 (Best)

| 模型 | Val Loss | Val PPL | 相对差距 |
|------|----------|---------|----------|
| **GPT** | **2.38** | **10.82** | 基准 |
| **OELM** | 2.61 | 13.54 | +25.1% |

### 3.3 关键观察

1. **收敛速度**: 两者收敛速度相近，都在Step 1000左右达到Val PPL < 50
2. **最终性能**: GPT的验证PPL比OELM优约25% (10.82 vs 13.54)
3. **参数量影响**: OELM参数量少7%，但性能下降约25%，需调整实现
4. **训练速度**: OELM训练略快 (Step 3800 vs Step 3200)

---

## 4. 结论与下一步

### 4.1 当前结论

- ✅ OELM在减少7%参数的情况下仍可收敛
- ⚠️ 当前OELM实现未启用冻结机制，性能有差距
- 🔧 需要实现正交矩阵冻结和梯度截断

### 4.2 下一步改进

1. **实现冻结机制**: 冻结Q/K投影矩阵，减少可训练参数
2. **调整学习率**: OELM可能需要更高的学习率
3. **增加冻结比例实验**: 测试freeze_ratio=0.25, 0.50等配置
4. **扩展到WikiText-103**: 在更大更复杂的数据集上验证

---

## 附录：实验复现命令

```bash
# 在MLDA GPU上运行
./mlda-run.sh train-both

# 查看状态
./mlda-run.sh status

# 查看日志
./mlda-run.sh logs
```

---

**实验时间**: 2026-02-06
**实验者**: 张天yu
**项目路径**: `~/Orthogonal_ELM_Transformers/Train`
