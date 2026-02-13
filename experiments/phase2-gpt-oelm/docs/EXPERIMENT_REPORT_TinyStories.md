# TinyStories 消融实验完整报告

> **实验阶段**: Phase 3 - GPT OELM 消融实验
> **数据集**: TinyStories
> **模型配置**: Medium-512 (d_model=512, n_layers=6, n_heads=8)
> **完成时间**: 2026-02-10

---

## 实验目标

验证分头正交初始化 (Head-wise Orthogonal Initialization) 在 GPT 语言模型上的有效性，通过完整的消融实验分析：
1. 冻结 Q/K 投影对性能的影响
2. 正交初始化 vs 随机初始化的效果
3. 与标准 Baseline 的性能差距

---

## 实验设计

### 三组实验对比

| 实验ID | 方法 | 初始化 | Q/K 状态 | 学习率 | 训练步数 |
|--------|------|--------|----------|--------|----------|
| GPT-01 | **Baseline** | 默认 Xavier | 可训练 | 3e-4 | 100K |
| GPT-02 | **OELM-Freeze** | 分头正交 | 冻结 | 1e-3 | 100K |
| GPT-03 | **OELM-Random** | 随机 Normal | 冻结 | 1e-3 | 100K |

### 关键变量控制
- 相同模型架构 (Medium-512)
- 相同数据集 (TinyStories)
- 相同 batch_size=8, seq_len=512
- 相同验证间隔 (每 1000 步)
- 相同随机种子

---

## 实验结果

### 最终性能对比

| 指标 | Baseline | OELM-Freeze | OELM-Random | 差距分析 |
|------|----------|-------------|-------------|----------|
| **Final PPL** | **4.27** ⭐ | **4.69** | **4.97** | Freeze +9.8%, Random +16.4% |
| Final Loss | 1.4516 | 1.5456 | 1.6038 | - |
| Best Val PPL | 4.27 | 4.69 | 4.97 | 相同 (最后即最佳) |
| **训练时间** | 4h 31m 27s | 4h 32m 46s | 4h 15m 18s | Random 快 6% |
| 平均步时间 | 0.0837s | 0.0835s | **0.0769s** | Random 快 8% |
| 总验证时间 | 7269s | 7393s | 7021s | - |

### 消融分析

```
性能差距分解:
├── Baseline → OELM-Freeze:  +9.8%  (冻结Q/K + 正交初始化)
├── Baseline → OELM-Random: +16.4%  (冻结Q/K + 随机初始化)
└── OELM-Random → OELM-Freeze: -6.0% (正交初始化的净提升)
```

**关键发现**:
1. **正交初始化有效**: 相比随机初始化，提升 6.0% PPL
2. **但冻结 Q/K 代价大**: 即使使用正交初始化，仍比 Baseline 差 9.8%
3. **训练速度**: OELM-Random 略快 (0.0769s vs 0.0837s/步)

---

## 详细分析

### 1. 验证 PPL 收敛曲线

| 训练阶段 | Baseline | OELM-Freeze | OELM-Random | 观察 |
|----------|----------|-------------|-------------|------|
| Step 10K | ~6.5 | ~7.2 | ~7.8 | Freeze/Random 起步较慢 |
| Step 30K | ~5.2 | ~5.6 | ~6.0 | 差距稳定 |
| Step 60K | ~4.5 | ~5.0 | ~5.3 | 差距约 +10-12% |
| Step 100K | **4.27** | **4.69** | **4.97** | 最终差距确认 |

### 2. 训练效率对比

| 效率指标 | Baseline | OELM-Freeze | OELM-Random | 结论 |
|----------|----------|-------------|-------------|------|
| 总训练时间 | 4h 31m | 4h 32m | 4h 15m | Random 略快 |
| 每步时间 | 0.0837s | 0.0835s | 0.0769s | 差距不大 |
| 显存使用 | 514M | 490M | 490M | Freeze 省 4.7% |
| 收敛速度 | 基准 | 较慢 | 最慢 | Freeze 需更长时间 |

### 3. 与 BERT/XNLI 对比

| 任务类型 | Baseline | OELM-Freeze | 结果 | 结论 |
|----------|----------|-------------|------|------|
| **BERT/XNLI** (分类) | 76.71% | **77.79%** ✅ | **+1.08%** | OELM 优于 Baseline |
| **GPT/TinyStories** (生成) | 4.27 PPL | 4.69 PPL ❌ | **+9.8%** | OELM 差于 Baseline |

**关键洞察**:
- OELM-Freeze 在 **分类任务 (BERT)** 上有效，但在 **生成任务 (GPT)** 上效果不佳
- 原因：语言模型对 Q/K 的依赖性比分类任务更强
- 生成任务需要更灵活的注意力机制来建模长程依赖

---

## 结论与讨论

### 主要结论

1. **正交初始化确实有效**
   - OELM-Freeze (4.69) 优于 OELM-Random (4.97)
   - 提升 6.0%，证明正交初始化比随机初始化更好

2. **但冻结 Q/K 对 GPT 代价太大**
   - OELM-Freeze 仍比 Baseline 差 9.8%
   - 超出 5% 的可用性目标范围
   - GPT 生成任务比 BERT 分类任务对 Q/K 更敏感

3. **训练速度优势不明显**
   - 每步时间差异 < 10%
   - 远小于 BERT/XNLI 上观察到的 57% 加速

### 局限性

1. **单一数据集**: 仅在 TinyStories 上测试，需在更大数据集验证
2. **单一模型规模**: 仅在 Medium-512 上测试
3. **超参数固定**: 学习率 1e-3 可能不是最优

### 未来方向

1. **部分解冻策略**
   - 每 N 步更新一次 Q/K
   - 渐进式解冻 (如训练后期解冻)

2. **更大规模验证**
   - OpenWebText (GPT-04/05)
   - WikiText-103 (GPT-06/07)

3. **超参数调优**
   - 尝试更小的学习率 (如 5e-4)
   - 调整 warmup 比例
   - 尝试不同的冻结策略

---

## 实验命令记录

### GPT-01 Baseline
```bash
CUDA_VISIBLE_DEVICES=2 python3 scripts/train_v2.py \
  --model_type baseline \
  --dataset tinystories \
  --max_lr 3e-4 \
  --max_steps 100000 \
  --out_dir outputs/GPT-01_baseline
```

### GPT-02 OELM-Freeze
```bash
CUDA_VISIBLE_DEVICES=3 python3 scripts/train_v2.py \
  --model_type oelm_v2 \
  --dataset tinystories \
  --max_lr 1e-3 \
  --max_steps 100000 \
  --out_dir outputs/GPT-02_oelm_freeze
```

### GPT-03 OELM-Random
```bash
CUDA_VISIBLE_DEVICES=2 python3 scripts/train_v2.py \
  --model_type oelm_random \
  --dataset tinystories \
  --max_lr 1e-3 \
  --max_steps 100000 \
  --out_dir outputs/GPT-03_oelm_random
```

---

## 附录：原始数据

### GPT-01 timing_stats.json
```json
{
  "model_type": "baseline",
  "final_perplexity": 4.269209111455862,
  "total_wall_time": 16287.72,
  "mean_step_time": 0.083659
}
```

### GPT-02 timing_stats.json
```json
{
  "model_type": "oelm_v2",
  "init_method": "orthogonal",
  "final_perplexity": 4.685668571064951,
  "total_wall_time": 16366.33,
  "mean_step_time": 0.083535
}
```

### GPT-03 timing_stats.json
```json
{
  "model_type": "oelm_random",
  "init_method": "normal",
  "final_perplexity": 4.9717059894364475,
  "total_wall_time": 15318.03,
  "mean_step_time": 0.076900
}
```

---

## 引用

如果本实验对您的研究有帮助，请引用：

```bibtex
@misc{oelm_gpt_2026,
  title={Orthogonal ELM Transformers: Head-wise Orthogonal Initialization for Efficient GPT Training},
  author={Research Team},
  year={2026},
  note={TinyStories Ablation Experiments}
}
```

---

> **报告生成时间**: 2026-02-10
> **实验负责人**: Research Team
> **服务器**: gpu43.dynip.ntu.edu.sg (4x RTX A5000)
