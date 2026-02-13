# Phase 2: GPT OELM 实验

> 将 BERT 上验证成功的 OELM 方法移植到 GPT 架构

---

## 实验目标

1. 将分头正交初始化从BERT移植到GPT
2. 解决之前全局正交失败的问题
3. 实现支持三种模式的统一训练框架
4. 为Phase 3消融实验提供技术基础

---

## 技术实现

### 核心创新: 分头正交

**问题**: 全局正交破坏多头结构
**解决**: 每个 head 独立 QR 分解

```python
def _init_orthogonal_heads(self):
    """分头正交初始化"""
    d_head = self.d_model // self.num_heads
    for h in range(self.num_heads):
        # 每个头单独正交初始化
        A = torch.randn(self.d_model, d_head)
        Q, R = torch.linalg.qr(A, mode='reduced')
```

### 三种模型类型

| 类型 | 初始化 | Q/K训练 | 用途 |
|------|--------|---------|------|
| `baseline` | 标准随机 | 可训练 | 基准对比 |
| `oelm_v2` | 分头正交 | 冻结 | 核心方法 |
| `oelm_random` | 标准随机 | 冻结 | 消融:验证正交价值 |

---

## 模型配置

### GPT Medium-512

| 配置 | 值 |
|------|-----|
| d_model | 512 |
| num_layers | 6 |
| num_heads | 8 |
| d_ff | 2048 |
| seq_len | 512 |
| 总参数量 | ~82M |

### 训练配置

| 配置 | Baseline | OELM-Freeze |
|------|----------|-------------|
| 学习率 | 3e-4 | 1e-3 |
| Batch Size | 8 | 8 |
| Max Steps | 100K | 100K |
| 优化器 | AdamW | AdamW |
| AMP | ✅ | ✅ |

---

## 实验结果

### TinyStories初步结果

| 指标 | Baseline | OELM-Freeze | 对比 |
|------|----------|-------------|------|
| Final PPL | 4.27 | 4.69 | **+9.8%** ❌ |
| 训练时间 | 4h 31m | 4h 32m | 相同 |
| 每步时间 | 0.184s | 0.184s | 相同 |

### 结果分析

**技术移植**: ✅ 成功
- 分头正交实现正确
- 训练流程稳定

**性能目标**: ❌ 未达成
- 目标: PPL ≤ 4.48 (Baseline × 1.05)
- 实际: PPL 4.69 (Baseline × 1.098)
- 差距: 超出5%目标

---

## 目录结构

```
phase2-gpt-oelm/
├── README.md              # 本文件
├── REPORT.md              # 详细实验报告
├── models/
│   ├── modeling_oelm_v2.py       # OELM模型(分头正交)
│   ├── modeling_gpt.py           # Baseline模型
│   ├── modeling_oelm.py          # 旧版(全局正交)
│   └── __init__.py
├── scripts/
│   ├── train_v2.py               # 主训练脚本
│   ├── run_phase2_experiments.sh # Phase 2启动
│   ├── run_phase3_experiments.sh # Phase 3启动
│   ├── start_gpt03.sh            # GPT-03启动
│   ├── start_gpt04.sh            # GPT-04启动
│   ├── start_gpt05.sh            # GPT-05启动
│   └── ... (其他工具脚本)
├── data/
│   └── prepare_data.py           # 数据准备
├── docs/
│   ├── EXPERIMENT_REPORT.md
│   ├── EXPERIMENT_REPORT.pdf
│   └── ... (其他文档)
├── checkpoints/             # 模型检查点
└── outputs/                 # 实验输出
    ├── GPT-01_baseline/
    └── GPT-02_oelm_freeze/
```

---

## 使用方法

### 启动Phase 2实验

```bash
# 在远程服务器上执行
cd ~/Orthogonal_ELM_Transformers/Train/experiments/phase2-gpt-oelm

# Phase 2实验
./scripts/run_phase2_experiments.sh

# 单独启动
./scripts/start_gpt03.sh  # OELM-Random消融
./scripts/start_gpt04.sh  # OpenWebText Baseline
./scripts/start_gpt05.sh  # OpenWebText OELM
```

### 训练脚本

```bash
# Baseline
python scripts/train_v2.py \
    --model_type baseline \
    --dataset tinystories \
    --max_lr 3e-4 \
    --max_steps 100000

# OELM-Freeze
python scripts/train_v2.py \
    --model_type oelm_v2 \
    --dataset tinystories \
    --max_lr 1e-3 \
    --max_steps 100000
```

---

## 关键发现

### 分头正交 vs 全局正交

| 方法 | PPL结果 | 状态 |
|------|---------|------|
| 全局正交 | +19% | ❌ 失败 |
| 分头正交 | +9.8% | ⚠️ 可用但不够理想 |

**结论**: 分头正交显著优于全局正交，但仍未达目标

### BERT vs GPT对比

| 架构 | 任务 | OELM效果 | 原因 |
|------|------|----------|------|
| BERT | 分类 | ✅ +1.08% | 注意力模式稳定 |
| GPT | 生成 | ❌ -9.8% | 需要动态Q/K调整 |

---

## 下一步

→ 详见 [`../phase3-gpt-ablation/`](../phase3-gpt-ablation/) - 全面消融实验

---

**实验完成时间**: 2026-02-09
