# Phase 3: GPT OELM 消融实验计划

## 实验目标

验证分头正交初始化 (Head-wise Orthogonal Initialization) 在 GPT 语言模型上的有效性。

**核心问题**: 冻结 Q/K 投影层并使用正交初始化，能否在保持性能的同时减少可训练参数？

**成功标准**: OELM-Freeze PPL ≤ Baseline PPL × 1.05

---

## 实验设计

### 数据集选择 (3个)

1. **TinyStories** (小规模验证)
   - 用途: 快速验证方法可行性
   - 规模: ~50M tokens
   - 训练步数: 100K
   - 预期时间: ~4.5小时

2. **OpenWebText** (中等规模真实数据)
   - 用途: 验证在真实Web文本上的效果
   - 规模: ~40B tokens
   - 训练步数: 150K
   - 预期时间: ~9小时

3. **WikiText-103** (标准基准)
   - 用途: 与已有研究对比
   - 规模: ~103M tokens
   - 训练步数: 200K
   - 预期时间: ~12小时

### 对比方法 (3个)

| 方法 | Q/K初始化 | Q/K训练 | 目的 |
|------|-----------|---------|------|
| **Baseline** | 标准随机 | 可训练 | 基准对比 |
| **OELM-Freeze** | 正交 | 冻结 | 核心方法验证 |
| **OELM-Random** | 随机 | 冻结 | 消融:验证正交init价值 |

---

## 实验矩阵

| 实验ID | 数据集 | 方法 | GPU | 学习率 | 步数 | 状态 |
|--------|--------|------|-----|--------|------|------|
| GPT-01 | TinyStories | Baseline | 2 | 3e-4 | 100K | ✅ 完成 |
| GPT-02 | TinyStories | OELM-Freeze | 3 | 1e-3 | 100K | ✅ 完成 |
| GPT-03 | TinyStories | OELM-Random | 2 | 1e-3 | 100K | ✅ 完成 |
| GPT-04 | OpenWebText | Baseline | 2 | 3e-4 | 150K | ✅ 完成 |
| GPT-05 | OpenWebText | OELM-Freeze | 3 | 1e-3 | 150K | ✅ 完成 |
| GPT-06 | WikiText-103 | Baseline | 2 | 3e-4 | 200K | ⏳ 待启动 |
| GPT-07 | WikiText-103 | OELM-Freeze | 3 | 1e-3 | 200K | ⏳ 待启动 |

---

## 启动脚本

### 单个实验

```bash
# TinyStories
cd experiments/phase3-gpt-ablation/scripts
./run_gpt01.sh [gpu_id]   # Baseline
./run_gpt02.sh [gpu_id]   # OELM-Freeze
./run_gpt03.sh [gpu_id]   # OELM-Random

# OpenWebText
./run_gpt04.sh [gpu_id]   # Baseline
./run_gpt05.sh [gpu_id]   # OELM-Freeze

# WikiText-103
./run_gpt06.sh [gpu_id]   # Baseline
./run_gpt07.sh [gpu_id]   # OELM-Freeze
```

### 批量启动

```bash
# 使用项目根目录的统一脚本
cd gpt-oelm-project
./scripts/run_phase3_experiments.sh GPT-01
./scripts/run_phase3_experiments.sh all   # 运行所有
```

---

## 监控方法

```bash
# 查看所有实验状态
cd experiments/common/scripts
./monitor_experiments.sh

# 实时监控 (5秒刷新)
./monitor_experiments.sh live

# 查看特定实验日志
./monitor_experiments.sh log GPT-04

# 直接查看tmux
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg "tmux capture-pane -t gpt04_openwebtext_baseline -p | tail -5"
```

---

## 结果分析

### 已完成的发现

#### TinyStories (✅ 完成)

| 指标 | Baseline | OELM-Freeze | OELM-Random |
|------|----------|-------------|-------------|
| Final PPL | 4.27 | 4.69 | 4.97 |
| vs Baseline | 基准 | +9.8% | +16.4% |

**关键发现**:
1. ✅ **正交初始化有效**: OELM-Freeze 比 OELM-Random 好 6.0%
2. ❌ **目标未达成**: +9.8% > 5% 目标
3. ⚠️ **冻结Q/K有代价**: 即使正交init也无法完全弥补

#### OpenWebText (✅ 完成)

| 指标 | Baseline | OELM-Freeze |
|------|----------|-------------|
| Final PPL | 47.24 | 54.29 |
| vs Baseline | 基准 | +14.9% |

**关键发现**:
1. ❌ **差距更大**: +14.9% > TinyStories (+9.8%)
2. ❌ **目标未达成**: 显著超出5%范围
3. 📊 **训练速度相同**: 0.184s/步 (无加速)

### 分析脚本

```bash
# 对比两个实验
cd experiments/common/scripts
python analyze_results.py \
  --exp1 ../../../gpt-oelm-project/outputs/GPT-01_baseline \
  --exp2 ../../../gpt-oelm-project/outputs/GPT-02_oelm_freeze

# 查看所有实验
python analyze_results.py --all
```

---

## 下一步计划

### 短期 (本周)

1. ✅ 完成 TinyStories 消融 (GPT-01/02/03)
2. ✅ 完成 OpenWebText 对比 (GPT-04/05)
3. ⏳ 启动 WikiText-103 (GPT-06/07)

### 中期 (下周)

4. 💡 分析 WikiText-103 结果
5. 💡 考虑改进策略:
   - 部分解冻 (只冻结部分层的Q/K)
   - 分层学习率
   - 渐进式解冻

### 长期

6. 📝 撰写阶段性报告
7. 🤔 评估是否继续冻结策略

---

## 文件位置

```
experiments/phase3-gpt-ablation/
├── PLAN.md                    # 本文件
├── scripts/
│   ├── run_gpt01.sh          # TinyStories Baseline
│   ├── run_gpt02.sh          # TinyStories OELM-Freeze
│   ├── run_gpt03.sh          # TinyStories OELM-Random
│   ├── run_gpt04.sh          # OpenWebText Baseline
│   ├── run_gpt05.sh          # OpenWebText OELM-Freeze
│   ├── run_gpt06.sh          # WikiText-103 Baseline
│   └── run_gpt07.sh          # WikiText-103 OELM-Freeze
├── configs/
│   ├── datasets.yaml         # 数据集配置
│   └── experiments.json      # 实验定义
└── results/                  # 结果汇总 (待创建)
```

---

## 参考文档

- 详细实验日志: `EXPERIMENT_LOG_Phase3.md`
- 项目主说明: `experiments/README.md`
- 训练脚本: `gpt-oelm-project/scripts/train_v2.py`
- 模型定义: `gpt-oelm-project/models/modeling_oelm.py`
