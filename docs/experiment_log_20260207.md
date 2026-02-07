# Q/K冻结实验日志

**实验名称**: Orthogonal ELM Transformer - Q/K Freeze Mechanism Study
**开始时间**: 2026-02-07 01:35
**记录时间**: 2026-02-07 14:15
**服务器**: MLDA GPU Cluster (gpu43.dynip.ntu.edu.sg)
**实验者**: 张天禹 (s125mdg43_10)

---

## 实验设计

### 三组对比

| 组别 | 模型 | freeze_qk | 可训练参数 | GPU分配 |
|------|------|-----------|------------|---------|
| **Group A** | GPT-Base | - | 44.9M (100%) | 0,1 |
| **Group B** | OELM-NoFreeze | False | 44.9M (100%) | 2,3 |
| **Group C** | OELM-Freeze | True | 41.8M (93%) | 🟢 运行中 |

### 固定参数

```yaml
n_layers: 6
d_model: 512
n_heads: 8
d_ff: 2048
seq_len: 512
batch_size: 8 (per GPU)
max_steps: 100000
learning_rate: 3e-4 (warmup 2K, cosine decay)
dataset: TinyStories
```

---

## 日志记录

### 2026-02-07 01:35 - 实验启动

- [x] 同步代码到服务器
- [x] 启动Group A (GPT-Base) - GPU 0,1
- [x] 启动Group B (OELM-NoFreeze) - GPU 2,3
- [x] 验证训练正常运行
- [ ] Group C (OELM-Freeze) 待启动

**启动命令记录**:
```bash
# Group A
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run --nproc_per_node=2 --master_port=29500 \
    scripts/02-训练脚本/train.py --model_type gpt ...

# Group B
export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.run --nproc_per_node=2 --master_port=29501 \
    scripts/02-训练脚本/train.py --model_type oelm --freeze_qk false ...
```

---

### 2026-02-07 02:13 - 第1次进度检查

**运行时长**: ~40分钟

#### Group A: GPT-Base

| 指标 | 数值 | 备注 |
|------|------|------|
| Step | 9,500 | 9.5% |
| Train Loss | 2.03 | 波动范围 1.76-2.19 |
| Train PPL | 7.62 | - |
| Best Val Loss | 1.8780 | @ Step 9,000 |
| **Best Val PPL** | **6.54** | 关键指标 |
| 学习率 | 2.96e-4 | 接近峰值 |

**收敛趋势**:
```
Step 1K:  Val PPL = 37.26
Step 2K:  Val PPL = 15.77
Step 6K:  Val PPL = 8.38
Step 9K:  Val PPL = 6.54 ✓
```

#### Group B: OELM-NoFreeze

| 指标 | 数值 | 备注 |
|------|------|------|
| Step | 11,800 | 11.8% |
| Train Loss | 1.98 | 波动范围 1.75-2.21 |
| Train PPL | 7.22 | - |
| Best Val Loss | 1.8390 | @ Step 11,000 |
| **Best Val PPL** | **6.29** | 关键指标 |
| 学习率 | 2.94e-4 | 接近峰值 |

**收敛趋势**:
```
Step 1K:  Val PPL = 36.65
Step 2K:  Val PPL = 16.03
Step 6K:  Val PPL = 8.20
Step 10K: Val PPL = 6.49
Step 11K: Val PPL = 6.29 ✓
```

#### 对比分析

| 对比项 | GPT | OELM-NoFreeze | 结论 |
|--------|-----|---------------|------|
| 当前Step | 9,500 | 11,800 | OELM快 **24%** |
| Best Val PPL | 6.54 | 6.29 | OELM优 **3.8%** |
| 收敛稳定性 | 良好 | 良好 | 相当 |
| Val PPL@9K | 6.54 | ~6.40 | 相近 |

**关键发现**:
1. ✅ OELM-NoFreeze训练速度比GPT快约24%
2. ✅ 两者Val PPL非常接近 (差距<4%)
3. ✅ 都成功收敛到6.x区间
4. ✅ 没有过拟合迹象

---

### 2026-02-07 14:00 - Group C 启动 (首次 - 4卡配置，后改为2卡)

**状态**: 🔄 已重启为2卡配置

**首次启动** (4卡 - 错误配置):
```bash
# 使用了4卡并行，不符合要求
GPU: 0,1,2,3 (4卡)
World size: 4
```

**重新启动** (2卡 - 正确配置):
```bash
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg '
screen -dmS exp_oelm_f bash -c "
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/usr1/home/s125mdg43_10/Orthogonal_ELM_Transformers/Train:$PYTHONPATH
source ~/projects/oelm/venv/bin/activate
cd /usr1/home/s125mdg43_10/Orthogonal_ELM_Transformers/Train
python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29502 \
    scripts/02-训练脚本/train.py \
    --model_type oelm --freeze_qk true \
    --d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 \
    --seq_len 512 --batch_size 8 --max_steps 100000 \
    --out_dir models/checkpoints/exp_oelm_freeze
"
'
```

**最终配置确认**:
| 参数 | 数值 |
|------|------|
| Model | OELM-Freeze |
| freeze_qk | **True** |
| 总参数 | 44,896,768 |
| 可训练参数 | 41,751,040 (**93.0%**) |
| 冻结参数 | 3,145,728 (**7.0%**) |
| **GPU** | **0,1 (2卡)** ✅ |
| **World size** | **2** ✅ |
| **Batch Size** | 8 per GPU |
| **有效Batch Size** | **16** |

**对比Group B的参数减少**:
- NoFreeze可训练: 44,896,768
- Freeze可训练: 41,751,040
- 减少: 3,145,728 (**7.0%**)

⚠️ **注意**: 实际减少7%而非预期的15%，这是因为Q/K矩阵仅占模型参数的一小部分。

---

### 2026-02-07 14:00 - 训练完成报告

**运行时长**: ~12.5小时

#### Group A: GPT-Base ✅ 完成

| 指标 | 数值 | 备注 |
|------|------|------|
| Step | **100,000** | 100% ✅ |
| Status | **Completed** | 正常结束 |
| Final Val Loss | 1.4793 | @ Step 100,000 |
| **Best Val Loss** | **1.4793** | @ Step 99,000 |
| **Best Val PPL** | **4.39** | 🏆 最佳结果 |
| 总参数 | 44.9M | 100% 可训练 |
| 训练时间 | ~12小时 | - |

**收敛曲线**:
```
Step 1K:   Val PPL = 37.26
Step 2K:   Val PPL = 15.77
Step 6K:   Val PPL = 8.38
Step 9K:   Val PPL = 6.54
Step 12K:  Val PPL = 5.95
Step 20K:  Val PPL = 5.30
Step 30K:  Val PPL = 4.97
Step 50K:  Val PPL = 4.65
Step 99K:  Val PPL = 4.39 ✓ Best
```

#### Group B: OELM-NoFreeze ✅ 完成

| 指标 | 数值 | 备注 |
|------|------|------|
| Step | **100,000** | 100% ✅ |
| Status | **Completed** | 正常结束 |
| Final Val Loss | 1.4857 | @ Step 100,000 |
| **Best Val Loss** | **1.4857** | @ Step 99,000 |
| **Best Val PPL** | **4.42** | 优秀结果 |
| 总参数 | 44.9M | 100% 可训练 |
| freeze_qk | False | Q/K可训练 |
| 训练时间 | ~12小时 | - |

**收敛曲线**:
```
Step 1K:   Val PPL = 36.65
Step 2K:   Val PPL = 16.03
Step 6K:   Val PPL = 8.20
Step 11K:  Val PPL = 6.29
Step 20K:  Val PPL = 5.42
Step 30K:  Val PPL = 5.05
Step 50K:  Val PPL = 4.71
Step 99K:  Val PPL = 4.42 ✓ Best
```

#### 最终对比分析

| 对比项 | GPT-Base | OELM-NoFreeze | 差距 |
|--------|----------|---------------|------|
| **Best Val PPL** | **4.39** | **4.42** | GPT优 **0.7%** |
| Final Val Loss | 1.4793 | 1.4857 | GPT优 0.4% |
| 训练步数 | 100K | 100K | 相同 |
| 总参数 | 44.9M | 44.9M | 相同 |
| 收敛速度 | 良好 | 良好 | 相当 |

**最终结论**:
1. ✅ **两模型性能几乎相同** - Val PPL差距仅0.7% (4.39 vs 4.42)
2. ✅ **OELM-NoFreeze验证成功** - 正交初始化+可训练Q/K达到标准Transformer性能
3. ✅ **无过拟合** - 验证损失持续下降至训练结束
4. ✅ **训练稳定** - 两实验均顺利完成100K步训练
5. 📋 **准备启动Group C** - 现在可启动OELM-Freeze对比组

---

## GPU监控记录

### 2026-02-07 02:13 状态

| GPU | 温度 | 功耗 | 利用率 | 显存使用 | 状态 |
|-----|------|------|--------|----------|------|
| 0 | 82°C | 196W | 100% | 23.0GB | 正常 |
| 1 | 80°C | 192W | 100% | 22.7GB | 正常 |
| 2 | 81°C | 194W | 99% | 21.4GB | 正常 |
| 3 | 81°C | 192W | 99% | 20.8GB | 正常 |

**温度评估**: 80-82°C属于正常工作范围，无需干预。

---

## 预计时间表

| 实验组 | 状态 | 最佳Val PPL | 完成时间 | 备注 |
|--------|------|-------------|----------|------|
| Group A | ✅ **已完成** | **4.39** | 2026-02-07 14:00 | 100K步, 2卡 |
| Group B | ✅ **已完成** | **4.42** | 2026-02-07 14:00 | 100K步, 2卡 |
| Group C | 🟢 **运行中** | - | 预计2026-02-08 02:00 | **2卡并行**, Step 100+ |

---

## 待办事项

### 高优先级
- [x] Group A/B训练完成
- [x] **启动Group C (OELM-Freeze)** ✅ 2026-02-07 14:00
- [ ] 监控Group C训练进度 ⬅️ 当前任务
- [ ] 下载训练日志备份

### 中优先级
- [ ] 等待Group C完成
- [ ] 准备三组对比分析

### 低优先级
- [ ] 生成可视化图表
- [ ] 准备论文图表

---

## 文件路径

### 本地
```
/Users/starryyu/Library/Mobile Documents/com~apple~CloudDocs/学术研究/Orthogonal ELM Transformers/Train/
├── docs/experiment_log_20260207.md (本文件)
├── docs/experiment_plan_qk_freeze.md
├── docs/phase2_completion_report.md
├── docs/phase3_completion_report.md
├── docs/phase4_experiment_status.md
├── scripts/experiment_qk_freeze.py
├── scripts/analyze_freeze_experiment.py
└── scripts/start_exp_c.sh
```

### 服务器
```
~/Orthogonal_ELM_Transformers/Train/
├── models/checkpoints/exp_gpt_base/
│   ├── training.log
│   └── best_model.pt
├── models/checkpoints/exp_oelm_no_freeze/
│   ├── training.log
│   └── best_model.pt
└── models/checkpoints/exp_oelm_freeze/ (待创建)
```

---

## 监控命令备忘

```bash
# 查看实时日志 - Group A
tail -f ~/Orthogonal_ELM_Transformers/Train/models/checkpoints/exp_gpt_base/training.log

# 查看实时日志 - Group B
tail -f ~/Orthogonal_ELM_Transformers/Train/models/checkpoints/exp_oelm_no_freeze/training.log

# 查看GPU状态
nvidia-smi

# 查看进程
ps aux | grep train.py

# 查看screen会话
screen -ls
screen -r exp_gpt
screen -r exp_oelm_nf
```

---

## 启动Group C命令

当Group A/B达到Step 50K时执行:

```bash
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg

# 方法1: 使用脚本
cd ~/Orthogonal_ELM_Transformers/Train
./scripts/start_exp_c.sh

# 方法2: 手动启动
screen -dmS exp_oelm_f bash -c "
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=~/Orthogonal_ELM_Transformers/Train:$PYTHONPATH
source ~/projects/oelm/venv/bin/activate
cd ~/Orthogonal_ELM_Transformers/Train
python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=29502 \
    scripts/02-训练脚本/train.py \
    --model_type oelm --freeze_qk true \
    --d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 \
    --seq_len 512 --batch_size 8 --max_steps 100000 \
    --data_path data/tiny_stories/train.bin \
    --out_dir models/checkpoints/exp_oelm_freeze \
    2>&1 | tee models/checkpoints/exp_oelm_freeze/training.log
exec bash
"
```

---

## 附录

### 假设验证检查表

| 假设 | 验证方法 | 当前状态 |
|------|----------|----------|
| H1: Freeze参数减少15% | 对比B和C的参数统计 | ❌ **未通过** (实际7%) |
| H2: Freeze与NoFreeze PPL差距<5% | 对比Val PPL | ⏳ 待Group C完成 |
| H3: Freeze速度>NoFreeze | 对比训练速度 | ⏳ 待Group C完成 |
| H4: Freeze性能接近GPT | 对比Val PPL | ⏳ 待Group C完成 |

### 问题记录

| 时间 | 问题 | 解决方案 | 状态 |
|------|------|----------|------|
| 01:35 | PYTHONPATH未设置 | 在启动命令中添加 | 已解决 |
| 01:36 | --val_data_path参数错误 | 移除该参数 | 已解决 |

---

**记录者**: Claude Code AI Assistant
**更新时间**: 2026-02-07 14:10
