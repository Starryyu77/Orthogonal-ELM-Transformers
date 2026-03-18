# OELM 预训练验证实验计划

**创建时间**: 2025年3月19日  
**执行地点**: NTU EEE GPU Cluster (2× Pro 6000)  
**实验状态**: 🟡 准备中

---

## 🎯 核心目标

验证**正交初始化 + Q/K冻结**在**预训练阶段**能否让模型学到更好的通用表示，从而在下游任务（不微调/少微调）时表现更优。

> **关键问题**: 微调场景下QK冻结无优势，预训练场景是否有效？

---

## 📊 实验阶段总览

| 阶段 | 名称 | 数据集 | 规模 | 预计时间 | GPU需求 | 目的 |
|:---:|:-----|:-------|:-----|:---------|:--------|:-----|
| **1** | 快速验证 | TinyStories | 2GB | 3-4小时 | 1× GPU | 验证QK冻结不阻碍学习 |
| **2** | 快速评估 | SST-2, AG News (子集) | 5K样本 | 1-2小时 | 1× GPU | 评估表示质量 |
| **3** | 中等预训练 | OpenWebText | 40GB | 3-5天 | 2× GPU (DDP) | 真实规模对比 |
| **4** | 全面评估 | GLUE多项任务 | 完整数据集 | 2-3天 | 1× GPU | 系统验证效果 |
| **5** | 大规模验证 | C4子集 | 100GB | 1-2周 | 2× GPU (DDP) | 可扩展性测试 |

**当前进度**: ⏳ 等待阶段1开始

---

## 🔬 实验组配置

### 对比方法

| 方法 | 可训练参数 | 策略 | 学习率 | 批次大小 |
|:-----|:-----------|:-----|:-------|:---------|
| **Baseline** | 100% (124M) | 标准预训练 | 5e-4 → 3e-4 | 256-512 |
| **OELM-QK** | ~75% (95M) | 冻结Query/Key投影 | 5e-4 → 3e-4 | 256-512 |
| **OELM-QK-FFN** | ~65% (81M) | 冻结Q/K + FFN层 | 5e-4 → 3e-4 | 256-512 |

### 参数冻结策略

```python
# OELM-QK: 冻结Q/K投影
for name, param in model.named_parameters():
    if 'q_proj' in name or 'k_proj' in name or 'W_q' in name or 'W_k' in name:
        param.requires_grad = False

# OELM-QK-FFN: 额外冻结FFN
for name, param in model.named_parameters():
    if 'ffn' in name or 'up_proj' in name or 'down_proj' in name:
        param.requires_grad = False
```

---

## 📋 各阶段详细配置

### 阶段1: 快速验证 (TinyStories)

**目标**: 验证QK冻结在预训练阶段不会阻碍基础学习能力

**数据集**: TinyStories
- 规模: 2GB文本，~500K样本
- 内容: 简单英语故事（适合124M模型）
- 下载: `datasets.load_dataset("roneneldan/TinyStories")`

**配置**:
```yaml
模型: GPT-2 small (124M参数, d_model=768, n_layer=12, n_head=12)
训练步数: 10,000 steps (~3个epoch)
Batch size: 32 per GPU × 2 GPUs = 64 total
序列长度: 512 tokens
优化器: AdamW (β1=0.9, β2=0.95, weight_decay=0.1)
学习率: 5e-4 (constant with linear warmup)
Warmup steps: 1,000
梯度裁剪: 1.0
保存频率: 每2,000 steps
验证频率: 每500 steps

实验组:
  - Baseline: 全可训练
  - OELM-QK: freeze_qk=true
  - OELM-QK-FFN: freeze_qk=true, freeze_ffn=true
```

**成功标准**:
- [ ] 所有组Loss稳定下降
- [ ] 验证PPL差距 < 5%
- [ ] 训练速度差异 < 20%

**预计时间**: 3-4小时 (2× Pro 6000)

---

### 阶段2: 快速评估 (下游任务)

**目标**: 评估预训练模型的表示质量（冻结backbone）

**评估方法**: Linear Probe
```python
# 1. 加载预训练模型，冻结所有参数
model = load_pretrained(checkpoint)
for param in model.parameters():
    param.requires_grad = False

# 2. 只训练轻量级分类头 + LayerNorm
classifier = nn.Sequential(
    nn.LayerNorm(768),
    nn.Linear(768, num_classes)
)
```

**数据集** (简化版，快速训练):
| 数据集 | 原始大小 | 使用子集 | 训练时间 |
|:-------|:---------|:---------|:---------|
| SST-2 | 67K | 5K | ~10分钟 |
| AG News | 120K | 5K | ~10分钟 |

**配置**:
```yaml
训练步数: 500 steps (子集)
学习率: 1e-3
Batch size: 64
冻结策略: 完全冻结backbone
```

**对比指标**:
| 指标 | Baseline | OELM-QK | OELM-QK-FFN |
|------|----------|---------|-------------|
| SST-2 Acc | XX% | XX% | XX% |
| AG News Acc | XX% | XX% | XX% |

**决策点**:
- ✅ 如果 OELM >= Baseline: 进入阶段3
- ⚠️ 如果差距 < 2%: 调整参数后重试
- ❌ 如果差距 > 5%: 分析原因，修改冻结策略

**预计时间**: 1-2小时 (3 checkpoints × 2 datasets × 10分钟)

---

### 阶段3: 中等规模预训练 (OpenWebText)

**目标**: 在真实规模数据上验证OELM效果

**数据集**: OpenWebText
- 规模: 40GB文本，~800万文档
- 来源: Reddit高赞链接提取
- 下载: `datasets.load_dataset("openwebtext")`

**配置**:
```yaml
模型: GPT-2 small (124M参数)
训练步数: 100,000 steps (~1.5个epoch)
Batch size: 128 per GPU × 2 GPUs = 256 total
序列长度: 1024 tokens
优化器: AdamW (weight_decay=0.1)
学习率: 3e-4 (cosine decay to 1e-5)
Warmup steps: 5,000
梯度累积: 2 steps (有效batch=512)
梯度裁剪: 1.0

保存频率: 每10,000 steps
验证频率: 每5,000 steps
日志频率: 每100 steps

实验组:
  - Baseline: 全可训练
  - OELM-QK: freeze_qk=true
  - OELM-QK-FFN: freeze_qk=true, freeze_ffn=true
```

**关键监测指标**:
- 训练Loss曲线（对比3组）
- 验证PPL（OpenWebText验证集10%）
- 训练速度（tokens/sec per GPU）
- GPU显存使用
- 梯度范数（稳定性检查）

**预计时间**: 3-5天 (2× Pro 6000, DDP训练)

---

### 阶段4: 全面下游评估 (GLUE)

**目标**: 在多个下游任务上系统验证预训练效果

**评估策略**:

| 设置 | 冻结范围 | 可训练参数 | 适用场景 |
|:-----|:---------|:-----------|:---------|
| **Setting A** | Full backbone | ~0.1M (仅分类头) | 纯表示质量 |
| **Setting B** | Backbone + LN | ~0.5M (+ LayerNorm) | 表示+适应 |
| **Setting C** | 底部10层 | ~25M (顶层2层) | 轻量微调 |

**数据集** (完整GLUE子集):
| 数据集 | 任务类型 | 训练样本 | 测试样本 | 评估指标 |
|:-------|:---------|:---------|:---------|:---------|
| **SST-2** | 情感分析 | 67K | 1.8K | Accuracy |
| **AG News** | 主题分类 | 120K | 7.6K | Accuracy |
| **MNLI** | NLI | 392K | 9.8K | Accuracy |
| **QQP** | 重复检测 | 363K | 40K | Accuracy/F1 |
| **MRPC** | 句子相似 | 3.7K | 1.7K | F1 |

**配置**:
```yaml
训练步数: 3 epochs (各数据集不同)
学习率: 2e-5 (Setting A/B), 1e-4 (Setting C)
Batch size: 32-64 (根据数据集)
早停: Patience=3
```

**预期结果**:
- **Setting A (Zero-shot)**: OELM应显著优于Baseline
- **Setting B**: OELM应优于或等于Baseline
- **Setting C**: 差距缩小（微调弥补表示差距）

**预计时间**: 2-3天 (5 datasets × 3 settings × 3 checkpoints)

---

### 阶段5: 大规模验证 (C4子集)

**目标**: 验证OELM在超大规模数据上的可扩展性

**数据集**: C4-en (子集)
- 规模: 100GB文本
- 来源: Common Crawl清洗
- 下载: `datasets.load_dataset("c4", "en", streaming=True)`

**配置**:
```yaml
模型: GPT-2 small (124M) 或 medium (350M)
数据: C4-en前100GB
训练步数: 300,000 steps (~0.5个epoch)
Batch size: 128 per GPU × 2 GPUs = 256
序列长度: 1024
学习率: 2.5e-4 (cosine decay)
Warmup steps: 10,000

实验组 (只保留阶段3-4中最优的):
  - Baseline
  - OELM-QK (如果效果更好)
  - OELM-QK-FFN (如果效果更好)
```

**预计时间**: 1-2周 (2× Pro 6000)

---

## 🛠️ NTU Cluster 执行配置

### 硬件配置
```bash
# SLURM配置
#SBATCH --partition=cluster02
#SBATCH --gpus=pro-6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00  # 7天，阶段3-5需要
```

### 环境设置
```bash
# 登录
ssh tianyu016@10.97.216.128

# 进入项目目录
cd /projects/LlamaFactory/OELM-Pretrain

# 激活环境
conda activate oelm

# 检查GPU
nvidia-smi
```

### 目录结构
```
/projects/LlamaFactory/OELM-Pretrain/
├── outputs/
│   ├── stage1_tinystories/      # 阶段1输出
│   ├── stage3_openwebtext/      # 阶段3输出
│   └── stage5_c4/               # 阶段5输出
├── checkpoints/                 # 模型检查点
│   ├── baseline/
│   ├── oelm_qk/
│   └── oelm_qk_ffn/
├── logs/                        # 训练日志
└── results/                     # 评估结果
```

---

## 📈 进度跟踪

| 阶段 | 状态 | 开始时间 | 完成时间 | 最佳结果 |
|:-----|:-----|:---------|:---------|:---------|
| 1. TinyStories 预训练 | ⏳ 待开始 | - | - | - |
| 2. 快速评估 | ⏳ 待开始 | - | - | - |
| 3. OpenWebText 预训练 | ⏳ 待开始 | - | - | - |
| 4. GLUE全面评估 | ⏳ 待开始 | - | - | - |
| 5. C4大规模验证 | ⏳ 待开始 | - | - | - |

**下次更新**: 阶段1完成后

---

## 🎯 预期成果

### 成功标准
1. **阶段1**: PPL差距 < 5%，训练稳定
2. **阶段3**: PPL差距 < 3%，速度提升 > 10%
3. **阶段4 (Setting A)**: OELM准确率 > Baseline + 3%

### 论文级贡献
- 证明预训练阶段QK冻结的有效性
- 提供参数-性能权衡的系统分析
- 开源可复现的代码和脚本

---

## 📞 联系信息

- **执行人**: tianyu016@ntu.edu.sg
- **服务器**: NTU EEE GPU Cluster (10.97.216.128)
- **GitHub**: https://github.com/Starryyu77/Orthogonal-ELM-Transformers

---

**最后更新**: 2025年3月19日  
**下次更新**: 阶段1完成后
