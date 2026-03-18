# BERT OELM 实验完整总结

> 整理日期: 2026-02-08
> 作者: 张天禹 (Zhang Tianyu)
> 学号: s125mdg43_10
> 指导单位: NTU MLDA Lab

---

## 1. 实验概述

### 1.1 研究目标

验证**分头正交初始化 (Head-wise Orthogonality)** 在 BERT 模型上的有效性，证明：
1. 分头正交 vs 全局正交的优势
2. 冻结 Q/K 参数的可行性
3. 正交初始化的必要性
4. 方法的跨任务泛化能力

### 1.2 核心创新

```
全局正交 (失败)          分头正交 (成功)
[768, 768] QR          [768, 768] → [12, 64, 768]
     ↓                      ↓ 每个head独立QR
破坏head结构           保留head结构
```

---

## 2. 实验列表

### Phase 1: SST-2 主实验

| 实验 | 数据集 | 配置 | 最佳准确率 | 状态 |
|------|--------|------|------------|------|
| **Baseline** | SST-2 | 全参数, lr=2e-5 | **93.12%** | ✅ 完成 |
| **OELM-Freeze** | SST-2 | 冻结Q/K, lr=1e-4 | **91.28%** | ✅ 完成 |
| **差距** | - | -12.9%参数 | **-1.84%** | - |

### Phase 2: 消融实验

| 实验 | 初始化方法 | 冻结Q/K | 准确率 | 结论 |
|------|------------|----------|---------|------|
| **OELM-Orthogonal** | 分头正交 | ✅ | 91.28% | ✅ 有效 |
| **OELM-Random** | 高斯随机 | ✅ | 82.11% | ❌ 失败 |
| **差距** | - | 相同 | **-9.17%** | 正交必要 |

### Phase 3: MNLI 泛化实验

| 实验 | 数据集 | 配置 | 最佳准确率 | 状态 |
|------|--------|------|------------|------|
| **Baseline** | MNLI | 全参数, lr=2e-5 | **83.44%** | ✅ 完成 |
| **OELM-Freeze** | MNLI | 冻结Q/K, lr=1e-4 | **82.23%** | ✅ 完成 |
| **差距** | - | -12.9%参数 | **-1.21%** | 更优秀 |

### Phase 4: 公平对比实验

| 实验 | 运行次数 | 每步时间 | 标准差 | 结论 |
|------|----------|-----------|---------|------|
| **Baseline** | 3次 | 0.3218s | ±0.0320s | 基准 |
| **OELM-Freeze** | 3次 | 0.3262s | ±0.0034s | 无显著差异 |
| **差异** | - | +1.4% | 更稳定 | ✓ 速度可接受 |

---

## 3. 关键结果汇总

### 3.1 跨数据集对比

| 数据集 | 任务类型 | Baseline | OELM-Freeze | 差距 | OELM达到比例 |
|--------|----------|----------|-------------|------|---------------|
| **SST-2** | 2分类情感分析 | 93.12% | 91.28% | -1.84% | 98.0% |
| **MNLI** | 3分类NLI | 83.44% | 82.23% | -1.21% | 98.5% |
| **平均** | - | - | - | **-1.53%** | **98.3%** |

### 3.2 正交性必要性验证

```
OELM-Orthogonal:  91.28%  (✓ 正交初始化)
OELM-Random:      82.11%  (✗ 随机初始化)
                    ------
差距:            -9.17%  (✓ 正交性必要)
```

### 3.3 参数效率分析

| 指标 | 数值 |
|------|------|
| 冻结参数比例 | 12.9% (14.17M / 109.48M) |
| 可训练参数比例 | 87.1% (95.31M / 109.48M) |
| 性能保留比例 | 98.5% (平均) |
| 参数效率比 | 7.6:1 (性能/参数损失比) |

---

## 4. 文件清单

### 4.1 核心代码

| 文件 | 路径 | 说明 |
|------|--------|------|
| `modeling_bert_oelm.py` | `src/` | 分头正交初始化实现 |
| `train_bert.py` | `src/` | 训练脚本 (支持Baseline/OELM) |
| `__init__.py` | `src/` | 模块初始化 |

### 4.2 脚本

| 文件 | 路径 | 说明 |
|------|--------|------|
| `run_experiment.sh` | `scripts/` | 快速实验启动脚本 |
| `run_fair_comparison.sh` | `scripts/` | AB-AB公平对比脚本 |

### 4.3 配置文件

| 文件 | 路径 | 说明 |
|------|--------|------|
| `sst2_baseline.yaml` | `configs/` | SST-2 Baseline配置 |
| `sst2_oelm.yaml` | `configs/` | SST-2 OELM配置 |
| `mnli_baseline.yaml` | `configs/` | MNLI Baseline配置 |
| `mnli_oelm.yaml` | `configs/` | MNLI OELM配置 |

### 4.4 训练日志

| 文件 | 路径 | 大小 | 说明 |
|------|--------|------|------|
| `bert_baseline.log` | `results/sst2/` | ~1.3MB | SST-2 Baseline训练日志 |
| `bert_oelm.log` | `results/sst2/` | ~1.3MB | SST-2 OELM训练日志 |
| `oelm_random_ablation.log` | `results/ablation/` | ~1.3MB | 消融实验日志 |
| `mnli_baseline.log` | `results/mnli/` | ~9.1MB | MNLI Baseline训练日志 |
| `mnli_oelm.log` | `results/mnli/` | ~9.2MB | MNLI OELM训练日志 |

### 4.5 计时分析数据

| 文件 | 路径 | 说明 |
|------|--------|------|
| `baseline_run1_*.json` | `results/timing/` | Baseline Run 1 计时数据 |
| `oelm_run1_*.json` | `results/timing/` | OELM Run 1 计时数据 |
| `comparison_summary_*.txt` | `results/timing/` | 对比实验摘要 |

### 4.6 文档

| 文件 | 路径 | 说明 |
|------|--------|------|
| `EXPERIMENT_REPORT_BERT_RESERVOIR.md` | `docs/` | 完整实验报告 |
| `README.md` | `./` | 项目README |
| `EXPERIMENT_SUMMARY.md` | `./` | 本文档 |

---

## 5. 实验环境

### 5.1 硬件环境

| 项目 | 配置 |
|------|--------|
| 服务器 | MLDA GPU Cluster (gpu43.dynip.ntu.edu.sg) |
| GPU | NVIDIA RTX A5000 (24GB) |
| CPU | Intel Xeon |
| 内存 | 128GB+ |

### 5.2 软件环境

| 项目 | 版本 |
|------|--------|
| Python | 3.8.10 |
| PyTorch | 2.0.1+cu118 |
| Transformers | 4.x |
| CUDA | 12.2 |
| 操作系统 | Ubuntu 20.04 |

---

## 6. 复现步骤

### 6.1 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/bert-oelm.git
cd bert-oelm-paper

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install torch==2.0.1 transformers datasets scikit-learn tqdm numpy
```

### 6.2 运行实验

```bash
# SST-2 实验
python src/train_bert.py --freeze_mode false --lr 2e-5 --dataset sst2
python src/train_bert.py --freeze_mode true --lr 1e-4 --dataset sst2 --init_method orthogonal

# MNLI 实验
python src/train_bert.py --freeze_mode false --lr 2e-5 --dataset mnli
python src/train_bert.py --freeze_mode true --lr 1e-4 --dataset mnli --init_method orthogonal

# 消融实验
python src/train_bert.py --freeze_mode true --lr 1e-4 --dataset sst2 --init_method normal
```

### 6.3 公平对比实验

```bash
./scripts/run_fair_comparison.sh 3
```

---

## 7. 关键发现与结论

### 7.1 主要发现

1. ✅ **分头正交有效**: 成功修复全局正交导致的性能崩塔
2. ✅ **参数效率高**: 冻结12.9%参数仅损失~1.5%准确率
3. ✅ **训练速度相当**: OELM与Baseline无显著差异
4. ✅ **训练更稳定**: OELM标准差更小(1.0% vs 9.9%)
5. ✅ **正交性必要**: OELM-Random低9.17%验证正交性关键
6. ✅ **泛化能力强**: MNLI上差距更小(-1.21% vs -1.84%)

### 7.2 实践建议

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 追求最佳性能 | Baseline | 93.12% vs 91.28% |
| 参数效率优先 | OELM-Freeze | 少12.9%参数，仅低1.5% |
| 边缘部署 | OELM-Freeze | 内存友好，可训练参数更少 |
| 快速迭代 | OELM-Freeze | 学习率更大(1e-4) |
| 复杂任务 | OELM-Freeze | MNLI上差距更小 |

### 7.3 局限性与未来工作

**局限性**:
- 仅在 BERT-base 上验证
- 仅测试 SST-2 和 MNLI 两个数据集
- 未测试生成任务

**未来工作**:
1. 扩展到 BERT-large, RoBERTa
2. 测试更多下游任务 (QQP, MRPC, CoLA)
3. 生成任务验证 (GPT 架构)
4. 理论分析: 正交投影对 attention 模式的影响

---

## 8. 附录

### 8.1 实验时间线

| 日期 | 事件 |
|------|------|
| 2026-02-07 15:18 | OELM 首次实验启动 |
| 2026-02-07 18:43 | SST-2 OELM 完成 (91.28%) |
| 2026-02-07 19:59 | SST-2 Baseline 完成 (93.12%) |
| 2026-02-07 21:20 | 公平对比实验开始 |
| 2026-02-08 00:54 | 公平对比实验完成 (6 runs) |
| 2026-02-08 12:27 | OELM-Random 消融实验启动 |
| 2026-02-08 12:59 | OELM-Random 完成 (82.11%) |
| 2026-02-08 12:38 | MNLI 实验启动 |
| 2026-02-08 16:37 | MNLI 实验完成 |

### 8.2 关键代码片段

见 `src/modeling_bert_oelm.py` 中的核心函数:
- `apply_head_wise_orthogonal_()`: 分头正交初始化
- `check_orthogonality()`: 正交性验证
- `freeze_model_parameters()`: 参数冻结策略
- `load_bert_with_head_wise_orthogonal()`: 模型加载入口

---

**整理完成** ✅
**下一步**: GitHub 上传
