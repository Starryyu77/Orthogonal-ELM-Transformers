# Phase 3: GPT OELM 消融实验日志

> 实验阶段: Phase 3
> 创建时间: 2026-02-09
> 最后更新: 2026-02-10
> 状态: 🟡 OpenWebText 完成, WikiText-103 待启动

---

## 实验目标

验证分头正交初始化 (Head-wise Orthogonal) 在 GPT 架构上的有效性：
- **TinyStories**: Baseline vs OELM-Freeze vs OELM-Random (完整消融)
- **OpenWebText**: Baseline vs OELM-Freeze
- **WikiText-103**: Baseline vs OELM-Freeze

成功标准: OELM-Freeze PPL ≤ Baseline PPL × 1.05

---

## 实验矩阵

| ID | 数据集 | 方法 | 学习率 | 步数 | 状态 | 启动时间 | 完成时间 |
|----|--------|------|--------|------|------|----------|----------|
| GPT-01 | TinyStories | Baseline | 3e-4 | 100K | ✅ 完成 | 2026-02-09 16:08 | 2026-02-09 20:40 |
| GPT-02 | TinyStories | OELM-Freeze | 1e-3 | 100K | ✅ 完成 | 2026-02-09 16:11 | 2026-02-09 20:44 |
| GPT-03 | TinyStories | OELM-Random | 1e-3 | 100K | ✅ 完成 | 2026-02-09 20:45 | 2026-02-10 01:06 |
| GPT-04 | OpenWebText | Baseline | 3e-4 | 150K | ✅ 完成 | 2026-02-10 14:15 | 2026-02-10 23:26 |
| GPT-05 | OpenWebText | OELM-Freeze | 1e-3 | 150K | ✅ 完成 | 2026-02-10 14:15 | 2026-02-10 23:26 |
| GPT-06 | WikiText-103 | Baseline | 3e-4 | 200K | ⏳ 未开始 | - | - |
| GPT-07 | WikiText-103 | OELM-Freeze | 1e-3 | 200K | ⏳ 未开始 | - | - |

---

## 模型配置 (Medium-512)

| 配置项 | 数值 |
|--------|------|
| d_model | 512 |
| num_layers | 6 |
| num_heads | 8 |
| d_ff | 2048 |
| seq_len | 512 |
| batch_size | 8 |
| vocab_size | 50257 |

---

## 启动记录

### GPT-01: TinyStories Baseline
```bash
# 2026-02-09 16:08
CUDA_VISIBLE_DEVICES=2 python3 scripts/train_v2.py \
  --model_type baseline \
  --dataset tinystories \
  --max_lr 3e-4 \
  --max_steps 100000 \
  --out_dir outputs/GPT-01_baseline
```

### GPT-02: TinyStories OELM-Freeze
```bash
# 2026-02-09 16:11
CUDA_VISIBLE_DEVICES=3 python3 scripts/train_v2.py \
  --model_type oelm_v2 \
  --dataset tinystories \
  --max_lr 1e-3 \
  --max_steps 100000 \
  --out_dir outputs/GPT-02_oelm_freeze
```

### GPT-03: TinyStories OELM-Random
```bash
# 2026-02-09 20:45
CUDA_VISIBLE_DEVICES=2 python3 scripts/train_v2.py \
  --model_type oelm_random \
  --dataset tinystories \
  --max_lr 1e-3 \
  --max_steps 100000 \
  --out_dir outputs/GPT-03_oelm_random
```

---

## 监控命令

```bash
# 查看训练进度
tmux capture-pane -t gpt01_baseline -p | tail -5
tmux capture-pane -t gpt02_oelm -p | tail -5

# 实时查看
tmux attach -t gpt01_baseline
tmux attach -t gpt02_oelm

# GPU 状态
nvidia-smi
```

---

## 输出位置

| 实验 | 日志 | 最佳模型 | 计时统计 |
|------|------|----------|----------|
| GPT-01 | `outputs/GPT-01_baseline/training.log` | `best.pt` | `timing_stats.json` |
| GPT-02 | `outputs/GPT-02_oelm_freeze/training.log` | `best.pt` | `timing_stats.json` |
| GPT-03 | `outputs/GPT-03_oelm_random/training.log` | `best.pt` | `timing_stats.json` |
| GPT-04 | `outputs/GPT-04_openwebtext_baseline/training.log` | `best.pt` | `timing_stats.json` |
| GPT-05 | `outputs/GPT-05_openwebtext_oelm/training.log` | `best.pt` | `timing_stats.json` |

---

## 结果汇总

### TinyStories 对比 (✅ 全部完成)

| 指标 | Baseline | OELM-Freeze | OELM-Random |
|------|----------|-------------|-------------|
| **Final PPL** | **4.27** | **4.69** | **4.97** |
| Final Loss | 1.4516 | 1.5456 | 1.6038 |
| 总训练时间 | 4h 31m 27s | 4h 32m 46s | 4h 15m 18s |
| 平均步时间 | 0.0837s | 0.0835s | 0.0769s |
| 与Baseline差距 | 基准 | **+9.8%** ⚠️ | **+16.4%** ❌ |

**关键发现**:
- OELM-Freeze 最终 PPL (4.69) 比 Baseline (4.27) 高 **9.8%**
- OELM-Random 最终 PPL (4.97) 比 Baseline 高 **16.4%**，比 OELM-Freeze 高 **6.0%**
- **正交初始化有效**: OELM-Freeze 比 OELM-Random 好 6.0%
- 但冻结 Q/K 本身导致性能下降，OELM-Freeze 仍超出 5% 目标
- 训练速度: OELM-Random 略快 (~0.077s/步 vs ~0.084s/步)

**消融分析**:
```
Baseline → OELM-Freeze:  +9.8% (冻结Q/K + 正交初始化)
Baseline → OELM-Random: +16.4% (冻结Q/K + 随机初始化)
OELM-Random → OELM-Freeze: -6.0% (正交初始化的提升)
```

### OpenWebText 对比 (✅ 全部完成)

| 指标 | Baseline | OELM-Freeze |
|------|----------|-------------|
| **Final PPL** | **47.24** | **54.29** |
| Best Val Loss | 3.855 | 3.994 |
| 总训练时间 | 9h 10m 18s | 9h 8m 46s |
| 平均步时间 | 0.184s | 0.184s |
| 与Baseline差距 | 基准 | **+14.9%** ❌ |

**关键发现**:
- OpenWebText 上 OELM-Freeze (54.29) 比 Baseline (47.24) 高 **14.9%**
- 差距比 TinyStories (+9.8%) 更大，说明大数据集上冻结 Q/K 代价更高
- 训练速度几乎相同 (0.184s/步)，OELM-Freeze 无显著加速

### WikiText-103 对比

| 指标 | Baseline | OELM-Freeze |
|------|----------|-------------|
| Best PPL | ⏳ 未开始 | ⏳ 未开始 |
| 训练时间 | - | - |

---

## 运行状态更新

### 2026-02-09 17:00 (运行中)

| 实验 | 进度 | 步数 | 已运行 | 剩余时间 | Loss | 状态 |
|------|------|------|--------|----------|------|------|
| GPT-01 | 12% | 12,494/100K | 29分钟 | ~3小时20分 | 1.87 | 🟢 正常 |
| GPT-02 | 11% | 11,288/100K | 26分钟 | ~3小时10分 | 2.14 | 🟢 正常 |

**观察**:
- 两个实验都在 GPU 2 和 3 上正常运行
- Loss 持续下降，训练稳定
- 预计完成时间: 2026-02-09 20:30 左右

### 2026-02-09 17:10 (运行中)

| 实验 | 进度 | 步数 | 已运行 | 剩余时间 | 当前Loss | 最佳Val PPL | 状态 |
|------|------|------|--------|----------|----------|-------------|------|
| GPT-01 | 21% | 21,001/100K | 51分钟 | ~1小时34分 | 1.84 | **5.65** | 🟢 正常 |
| GPT-02 | 20% | 20,001/100K | 48分钟 | ~1小时48分 | 2.08 | **6.35** | 🟢 正常 |

**验证结果对比**:
| 实验 | 验证步数 | Val Loss | Val PPL | 相对表现 |
|------|----------|----------|---------|----------|
| GPT-01 (Baseline) | 20,000 | 1.7315 | 5.65 | 基准 |
| GPT-02 (OELM-Freeze) | 19,000 | 1.8482 | 6.35 | +12.4% ⚠️ |

**观察**:
- 两个实验都完成了约20%的训练进度
- **GPT-01 Baseline PPL: 5.65** (更优)
- **GPT-02 OELM-Freeze PPL: 6.35** (比 Baseline 高 12.4%)
- 当前差距超出 5% 目标范围，需继续观察后续训练
- 两个实验预计还有 ~1.5 小时完成

**训练速度**:
- GPT-01: ~13.56 it/s
- GPT-02: ~12.33 it/s

### 2026-02-09 19:06 (运行中 - 接近完成)

| 实验 | 进度 | 步数 | 已运行 | 剩余时间 | 当前Loss | 最佳Val PPL | 状态 |
|------|------|------|--------|----------|----------|-------------|------|
| GPT-01 | 62% | 62,001/100K | 2小时57分 | ~44分钟 | 1.62 | **4.51** ✅ | 🟢 正常 |
| GPT-02 | 60% | 60,363/100K | 2小时54分 | ~47分钟 | 1.78 | **5.04** | 🟢 正常 |

**验证结果对比**:
| 实验 | 验证步数 | Val Loss | Val PPL | 与Baseline差距 |
|------|----------|----------|---------|----------------|
| GPT-01 (Baseline) | 61,000 | 1.5057 | **4.51** | 基准 |
| GPT-02 (OELM-Freeze) | 59,000 | 1.6176 | **5.04** | +11.8% ⚠️ |

**关键观察**:
- GPT-01 Baseline 在 Step 61000 达到 **PPL 4.51** (新最佳)
- GPT-02 OELM-Freeze 在 Step 59000 达到 **PPL 5.04**
- 差距为 +11.8%，仍然超出 5% 目标范围
- 两个实验预计 **40-50分钟后完成**
- 模型文件已保存: `best.pt` (514M for Baseline, 490M for OELM-Freeze)

**准备启动 GPT-03**:
- 脚本已准备: `scripts/start_gpt03.sh`
- 配置: TinyStories, OELM-Random, lr=1e-3, GPU 2
- 将在 GPT-01 完成后立即启动

---

### 2026-02-09 20:45 (GPT-01/02 完成, GPT-03 启动)

| 实验 | 状态 | 完成时间 | Final PPL | 训练时间 |
|------|------|----------|-----------|----------|
| GPT-01 Baseline | ✅ 完成 | 20:40:27 | **4.27** | 4h 31m |
| GPT-02 OELM-Freeze | ✅ 完成 | 20:44:31 | **4.69** | 4h 32m |
| GPT-03 OELM-Random | 🟢 运行中 | - | - | - |

**GPT-03 启动命令**:
```bash
./scripts/start_gpt03.sh
# 或
CUDA_VISIBLE_DEVICES=2 python3 scripts/train_v2.py \
  --model_type oelm_random \
  --dataset tinystories \
  --max_lr 1e-3 \
  --max_steps 100000 \
  --out_dir outputs/GPT-03_oelm_random
```

**性能差距分析**:
- OELM-Freeze PPL 4.69 vs Baseline PPL 4.27
- 差距: **+9.8%** (仍超出 5% 目标)
- 但比中期结果 (+11.8%) 有所改善
- 等待 GPT-03 结果判断正交初始化效果

---

### 2026-02-10 01:06 (✅ TinyStories 全部完成)

| 实验 | 状态 | 完成时间 | Final PPL | 训练时间 |
|------|------|----------|-----------|----------|
| GPT-01 Baseline | ✅ 完成 | 20:40:27 | **4.27** | 4h 31m |
| GPT-02 OELM-Freeze | ✅ 完成 | 20:44:31 | **4.69** | 4h 32m |
| GPT-03 OELM-Random | ✅ 完成 | 01:06:45 | **4.97** | 4h 15m |

**关键结论**:
1. **正交初始化有效**: OELM-Freeze (4.69) 优于 OELM-Random (4.97)，提升 6.0%
2. **但冻结 Q/K 代价大**: OELM-Freeze 仍比 Baseline 差 9.8%，超出 5% 目标
3. **GPT 上效果不如 BERT**: XNLI 上 OELM-Freeze 优于 Baseline (+1.08%)，但 GPT 上差距较大

**可能原因**:
- 语言模型 (GPT) 比分类任务 (BERT/XNLI) 对 Q/K 冻结更敏感
- 生成任务需要更灵活的注意力机制
- 学习率 1e-3 可能不够优化，需要更长的 warmup 或调整 schedule

---

### 2026-02-10 13:35 (🚀 准备启动 OpenWebText)

**状态更新**:
- ✅ OpenWebText 数据已准备完成 (189M train.bin, 2M val.bin)
- ✅ GPT-04/05 启动脚本已创建并同步到服务器
- ⏳ 等待 GPU 2/3 空闲 (当前被其他用户占用)

**启动脚本**:
```bash
# 手动启动 (GPU 空闲时)
./scripts/start_gpt04.sh  # GPU 2, Baseline, 150K steps
./scripts/start_gpt05.sh  # GPU 3, OELM-Freeze, 150K steps

# 或自动监控启动
./scripts/auto_start_openwebtext.sh
```

**预计训练时间**: 150K steps ≈ 6-7 小时

---

### 2026-02-10 14:15 (🚀 OpenWebText 实验已启动)

| 实验 | 状态 | 启动时间 | GPU | 进度 | 预计完成 |
|------|------|----------|-----|------|----------|
| GPT-04 Baseline | 🟢 运行中 | 14:15:08 | GPU 2 | 94/150K | ~21:00 |
| GPT-05 OELM-Freeze | 🟢 运行中 | 14:15:09 | GPU 3 | 98/150K | ~21:00 |

**监控命令**:
```bash
# 查看进度
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg "tmux capture-pane -t gpt04_openwebtext_baseline -p | tail -3"
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg "tmux capture-pane -t gpt05_openwebtext_oelm -p | tail -3"

# 实时查看
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg "tmux attach -t gpt04_openwebtext_baseline"
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg "tmux attach -t gpt05_openwebtext_oelm"
```

---

### 2026-02-10 23:26 (✅ OpenWebText 全部完成)

| 实验 | 状态 | 完成时间 | Final PPL | 训练时间 |
|------|------|----------|-----------|----------|
| GPT-04 Baseline | ✅ 完成 | 23:26:16 | **47.24** | 9h 10m |
| GPT-05 OELM-Freeze | ✅ 完成 | 23:26:30 | **54.29** | 9h 8m |

**OpenWebText 结果对比**:

| 指标 | Baseline | OELM-Freeze | 差距 |
|------|----------|-------------|------|
| Final PPL | 47.24 | 54.29 | +14.9% ❌ |
| Best Val Loss | 3.855 | 3.994 | +3.6% |
| 训练时间 | 9h 10m | 9h 8m | -0.4% |
| 平均步时间 | 0.184s | 0.184s | 0% |

**与 TinyStories 对比分析**:

| 数据集 | Baseline PPL | OELM-Freeze PPL | 差距 | 结论 |
|--------|--------------|-----------------|------|------|
| TinyStories | 4.27 | 4.69 | **+9.8%** ⚠️ | 接近目标 |
| OpenWebText | 47.24 | 54.29 | **+14.9%** ❌ | 超出目标 |

**关键结论**:
1. **大数据集上差距更大**: OpenWebText (+14.9%) > TinyStories (+9.8%)
2. **冻结 Q/K 对复杂任务影响更大**: 数据量增大时，固定注意力机制限制更明显
3. **训练速度无优势**: 平均步时间几乎相同 (0.184s)，无计算加速
4. **目标未达成**: 两次实验都超出 5% 目标范围

---

## 综合发现

### TinyStories 消融 (✅ 完成)
- OELM-Freeze vs Baseline: +9.8%
- OELM-Random vs Baseline: +16.4%
- **正交初始化有效**: OELM-Freeze 比 OELM-Random 好 6.0%

### OpenWebText 对比 (✅ 完成)
- OELM-Freeze vs Baseline: +14.9%
- **差距扩大**: 比 TinyStories 差 5.1 个百分点

### 现象解释
1. **语言模型对 Q/K 更敏感**: 比 BERT/XNLI (+1.08% 优势) 差距大得多
2. **数据复杂度影响**: OpenWebText 数据更多样，需要更灵活的注意力
3. **可能改进方向**:
   - 部分解冻 Q/K (如只解冻某些层)
   - 调整学习率 (当前 1e-3 可能过大或过小)
   - 增加 warmup 步数

---

## 下一步

1. 🚀 启动 WikiText-103 实验 (GPT-06, GPT-07) - 验证更大规模数据
2. 💡 考虑改进策略: 部分解冻、分层学习率
3. 📊 分析当前结果，撰写阶段性报告
4. 🤔 评估是否继续冻结策略或转向其他方向
