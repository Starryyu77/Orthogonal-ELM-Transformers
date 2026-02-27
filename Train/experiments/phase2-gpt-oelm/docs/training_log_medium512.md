# OELM Medium-512 训练日志

**实验日期**: 2026-02-06
**实验者**: 张天禹
**实验目标**: 对比标准GPT与OELM在Medium-512配置下的性能差异

---

## 1. 实验配置

### 1.1 模型架构
| 参数 | 值 |
|------|-----|
| Model Type | GPT vs OELM |
| n_layers | 6 |
| d_model | 512 |
| n_heads | 8 |
| d_ff | 2048 |
| seq_len | 512 |

### 1.2 训练参数
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

### 1.3 硬件环境
| 项目 | 配置 |
|------|------|
| 服务器 | MLDA GPU (NTU) |
| GPUs | 4 × NVIDIA RTX A5000 (24GB) |
| GPT分配 | GPU 0,1 |
| OELM分配 | GPU 2,3 |

---

## 2. 训练时间线

### 2.1 启动阶段 (15:30-16:30)

**15:30** - 开始配置环境
- ✅ 发现MLDA GPU上已有venv环境: `~/projects/oelm/venv`
- ✅ Python 3.8.10, PyTorch 2.0.1+cu118
- ✅ 复制现有数据: `~/projects/oelm/data/tinystories/`

**15:35** - 首次启动尝试
- ❌ 问题: `scripts/02-训练脚本/train.py` 路径错误
- ❌ 问题: `--val_data_path` 参数不被支持
- ❌ 问题: `--freeze_ratio` 参数不被支持
- ❌ 问题: WandB认证失败
- ❌ 问题: CUDA OOM (Batch Size 32)

**16:15** - 参数调整
- ✅ 移除 `--val_data_path`
- ✅ 移除 `--freeze_ratio` (OELM当前实现不支持)
- ✅ 禁用WandB (`--use_wandb` 已移除)
- ✅ Batch Size: 32 → 8

**16:31** - 训练成功启动
- ✅ GPT训练启动 (tmux: gpt_train)
- ✅ OELM训练启动 (tmux: oelm_train)
- ✅ 4块GPU全部100%运行

---

## 3. 训练过程记录

### 3.1 初始阶段 (Step 0-1000)

| Time | Step | GPT Loss | GPT PPL | OELM Loss | OELM PPL | Notes |
|------|------|----------|---------|-----------|----------|-------|
| 16:31 | 0 | 10.93 | 22026 | 10.91 | 22026 | 初始Loss相近 |
| 16:33 | 100 | 9.31 | 11102 | 9.41 | 12175 | GPT收敛略快 |
| 16:35 | 300 | 6.05 | 423.71 | 6.11 | 451.37 | 两者差距稳定 |
| 16:38 | 1000 | 3.75 | 42.55 | 3.85 | 46.90 | **首次验证** |
| - | - | **3.72** | **41.41** | **3.83** | **46.04** | Val PPL差距约11% |

**观察**:
- OELM在Step 0-1000阶段表现略逊于GPT
- 验证集PPL差距约11% (46.04 vs 41.41)
- 两者都成功收敛

### 3.2 中期阶段 (Step 1000-3800)

| Time | Step | GPT Loss | GPT PPL | OELM Loss | OELM PPL | Notes |
|------|------|----------|---------|-----------|----------|-------|
| 16:40 | 2000 | 3.05 | 21.09 | - | - | GPT继续下降 |
| 16:43 | 3000 | 2.54 | 12.68 | 2.80 | 16.37 | **Val Checkpoint** |
| - | - | **2.38** | **10.82** | **2.61** | **13.54** | 差距扩大到25% |
| 16:45 | 3200 | 2.53 | 12.60 | - | - | GPT训练Loss |
| 16:45 | 3800 | - | - | 2.50 | 12.13 | OELM训练Loss |

**关键发现**:
- **Step 3000验证**: GPT Val PPL=10.82, OELM Val PPL=13.54
- **性能差距**: OELM验证PPL比GPT高约25%
- **参数量**: OELM (41.8M) 比 GPT (44.9M) 少7%
- **训练速度**: OELM略快 (Step 3800 vs 3200)

---

## 4. 问题与解决

### 4.1 已解决的问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| CUDA OOM | Batch Size 32过大 | 调整为Batch Size 8 |
| 模块找不到 | PYTHONPATH未设置 | 添加`export PYTHONPATH=$PROJECT_DIR` |
| WandB认证失败 | 未登录 | 禁用WandB |
| 参数不支持 | 脚本版本不匹配 | 移除`--val_data_path`和`--freeze_ratio` |
| 数据缺失 | 未准备数据 | 复制现有数据`~/projects/oelm/data/` |

### 4.2 当前限制

| 限制 | 影响 | 优先级 |
|------|------|--------|
| OELM未启用冻结机制 | 失去架构特性，性能下降 | 🔴 高 |
| 仅在TinyStories测试 | 数据集过于简单 | 🟡 中 |
| 无冻结比例对比实验 | 无法验证ELM理论 | 🔴 高 |
| 训练时间有限 | 仅完成~4%总步数 | 🟢 低 |

---

## 5. 性能分析

### 5.1 当前性能对比 (截至Step 3800)

| 指标 | GPT | OELM | 差距 |
|------|-----|------|------|
| 总参数量 | 44.9M | 41.8M | -7.0% |
| 可训练参数 | 44.9M (100%) | 41.8M (100%) | -7.0% |
| 冻结参数 | 0 | 0 | - |
| Val Loss | **2.38** | 2.61 | +9.7% |
| Val PPL | **10.82** | 13.54 | +25.1% |
| 训练速度 | Step 3200 | Step 3800 | OELM +18% |

### 5.2 关键观察

1. **性能差距**: OELM验证PPL比GPT高25%，超出预期
2. **原因分析**:
   - 当前OELM实现未冻结Q/K矩阵，失去关键特性
   - 所有参数仍可训练，等同于标准Transformer
   - 正交初始化仅影响初始状态，无持续约束

3. **理论预期 vs 实际**:
   - 预期: 冻结50%注意力参数，性能下降<10%
   - 实际: 无冻结，性能下降25%
   - 结论: **必须实现冻结机制**

---

## 6. 后续实验计划

### 6.1 立即执行 (优先级: 🔴 高)

#### 任务1: 实现OELM冻结机制
- **目标**: 在`modeling_oelm.py`中实现Q/K矩阵冻结
- **方法**:
  - 添加`freeze_ratio`参数
  - 对Q/K投影矩阵`requires_grad=False`
  - 支持分层冻结策略
- **预期结果**: 减少可训练参数，验证ELM理论

#### 任务2: 冻结比例对比实验
- **配置**:
  - freeze_ratio=0.075 (仅Q/K, ~7.5%)
  - freeze_ratio=0.25 (Q/K+部分FFN, ~25%)
  - freeze_ratio=0.50 (Q/K+更多FFN, ~50%)
- **目标**: 找到性能-效率最佳平衡点

### 6.2 短期计划 (优先级: 🟡 中)

#### 任务3: WikiText-103数据集验证
- **原因**: TinyStories过于简单，缺乏说服力
- **方法**: 在更复杂的数据集上测试PPL
- **预期**: 验证OELM的通用语言建模能力

#### 任务4: 训练至收敛
- **当前进度**: ~4% (3800/100000 steps)
- **目标**: 完成100K steps训练
- **预计时间**: 10-12小时
- **检查点**: 每5000 steps验证并保存最佳模型

### 6.3 长期计划 (优先级: 🟢 低)

#### 任务5: 多尺寸对比
- Tiny (d_model=256, 4 layers)
- Small (d_model=384, 6 layers) ✅ 当前
- Medium (d_model=768, 12 layers)
- Large (d_model=1024, 24 layers)

#### 任务6: 下游任务评估
- 文本生成质量
- 零样本/少样本学习能力
- 与其他高效Transformer对比

---

## 7. 实验命令速查

```bash
# 查看训练状态
./mlda-run.sh status

# 查看训练日志
./mlda-run.sh logs

# 实时查看GPT日志
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg 'tail -f ~/Orthogonal_ELM_Transformers/Train/logs/gpt_train.log'

# 实时查看OELM日志
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg 'tail -f ~/Orthogonal_ELM_Transformers/Train/logs/oelm_train.log'

# 停止训练
./mlda-run.sh kill-all
```

---

## 8. 附录

### 8.1 模型检查点

| 模型 | 最佳Val Loss | 保存路径 |
|------|-------------|----------|
| GPT | 2.3812 | `models/checkpoints/gpt_medium512/best_model.pt` |
| OELM | 2.6056 | `models/checkpoints/oelm_medium512/best_model.pt` |

### 8.2 训练日志文件

| 文件 | 路径 |
|------|------|
| GPT日志 | `logs/gpt_train.log` |
| OELM日志 | `logs/oelm_train.log` |

### 8.3 相关文档

- 实验报告: `docs/experiment_report_medium512.md`
- 训练脚本: `mlda-run.sh`
- 模型实现: `models/modeling_oelm.py`, `models/modeling_gpt.py`

---

**日志更新时间**: 2026-02-06 16:45
**下次更新**: 达到Step 5000或遇到重大问题时
