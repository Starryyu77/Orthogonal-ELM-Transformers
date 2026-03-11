# OELM Pretrain 实验 - AI 交接文档

**创建时间**: 2026-03-09  
**最后更新**: 2026-03-11 02:10 UTC
**项目路径**: `/projects/LlamaFactory/OELM-Pretrain/`  
**集群**: NTU EEE GPU Cluster (10.97.216.128)  
**用户名**: tianyu016  

---

## 最新状态 (实时)

| 阶段 | 作业ID | 方法 | 状态 | 节点 | 备注 |
|------|--------|------|------|------|------|
| **预训练** | - | Baseline | ✅ **完成** | - | Loss: 0.8592, PPL: 2.36 |
| **预训练** | - | OELM-QK | ✅ **完成** | - | ~196000步 |
| **预训练** | - | OELM-QK-FFN | ✅ **完成** | - | ~216000步 |
| **微调** | 44374 | Baseline | 🟢 **Running** | gpu-pro6000-1 | IMDB情感分类 |
| **微调** | 44375 | OELM-QK | 🟢 **Running** | gpu-pro6000-1 | IMDB情感分类 |
| **微调** | 44376 | OELM-QK-FFN | ⏳ **Pending** | - | IMDB情感分类 |
|--------|------|------|------|------|----------|
| **44083** | **OELM-QK** | **🟢 Running** | gpu-pro6000-2 | 64.9% (86,003/132,483) | ~18:30 UTC |
| **44084** | **OELM-QK-FFN** | **🟢 Running** | gpu-pro6000-2 | 71.4% (94,586/132,483) | ~16:20 UTC |
|--------|------|------|------|----------|


### 已完成 ✅
1. **Baseline预训练**: ✅ 完成 (7小时34分钟, Final Loss: 0.8592, PPL: 2.36)
2. **OELM-QK预训练**: ✅ 完成 (~196000步, 超目标步数)
3. **OELM-QK-FFN预训练**: ✅ 完成 (~216000步, 超目标步数)
4. **磁盘清理**: 多次清理，释放~200GB空间
5. **微调启动**: 🟢 已提交3个IMDB微调作业

### 预训练结果
| 方法 | 总步数 | Loss | PPL | 训练时间 | 可训练参数 |
|------|--------|------|-----|----------|-----------|
| **Baseline** | 132,483 | 0.8592 | 2.36 | 7h34m | 100% (124.4M) |
| **OELM-QK** | ~196,000 | - | - | ~10h | 88.6% (110.2M) |
| **OELM-QK-FFN** | ~216,000 | - | - | ~10h | 43.1% (53.6M) |

### 微调配置
| 方法 | 数据集 | Epochs | 学习率 | 冻结Q/K | 预期时间 |
|------|--------|--------|--------|---------|----------|
| Baseline | IMDB | 3 | 3e-4 | ❌ | ~2h |
| OELM-QK | IMDB | 3 | 1e-3 | ✅ | ~2h |
| OELM-QK-FFN | IMDB | 3 | 1e-3 | ✅ | ~2h |

### 实验目标
**验证**: OELM方法在预训练+微调完整流程中的有效性
- **核心假设**: 更少参数的预训练模型能否达到与全参数Baseline相当的下游任务性能？
5. **微调启动**: 🟢 已提交3个IMDB微调作业
1. **Baseline预训练**: ✅ 已完成 (7小时34分钟, Final Loss: 0.8592, PPL: 2.36)
2. **磁盘清理**: 删除中间检查点，释放86.5GB空间
3. **OELM-QK/OELM-QK-FFN**: 🟢 训练中 (5小时+)

### 关键指标对比
| 方法 | Loss | PPL | 可训练参数 |
|------|------|-----|-----------|
| Baseline | 0.8592 | 2.36 | 100% |
| OELM-QK | 0.0294 | 1.03 | 88.6% |
| OELM-QK-FFN | 0.9391 | 2.56 | 43.1% |
### 遇到的问题与解决 ✅
1. **GPU配额限制**: 将脚本从 `--gpus=pro6000:2` 改为 `--gpus=pro6000:1`
2. **ImportError**: 修复 `CosineLRScheduler` → `LambdaLR`
3. **DataLoader错误**: 修复collate_fn resize问题
4. **磁盘空间不足**: 清理Baseline中间检查点，释放86.5GB
2. **ImportError**: 修复 `from torch.optim.lr_scheduler import CosineLRScheduler` → `LambdaLR`

### 监控命令
```bash
ssh tianyu016@10.97.216.128

# 查看作业状态
watch squeue -u tianyu016

# 查看最新训练进度 (OELM-QK: 44083)
tail -c 300 /projects/LlamaFactory/OELM-Pretrain/logs/pretrain_oelm_qk-44083.err | strings | tail -3

# 查看最新训练进度 (OELM-QK-FFN: 44084)
tail -c 300 /projects/LlamaFactory/OELM-Pretrain/logs/pretrain_oelm_qk_ffn-44084.err | strings | tail -3

# 查看检查点
ls -t /projects/LlamaFactory/OELM-Pretrain/outputs/pretrain/*/checkpoint-*/
```

### 预训练完成后运行下游微调
```bash
cd /projects/LlamaFactory/OELM-Pretrain
./scripts/submit_all_finetune.sh
```
```bash
ssh tianyu016@10.97.216.128
tail -f /projects/LlamaFactory/OELM-Pretrain/logs/pretrain_baseline-43715.out
```

---
# OELM Pretrain 实验 - AI 交接文档

**创建时间**: 2026-03-09  
**项目路径**: `/projects/LlamaFactory/OELM-Pretrain/`  
**集群**: NTU EEE GPU Cluster (10.97.216.128)  
**用户名**: tianyu016  

---

## 1. 我们在做什么事情

### 核心目标
验证 **OELM (Orthogonal ELM)** 方法在**预训练阶段**的有效性。

### 背景
- **Phase 1-4**: OELM在**微调分类任务**上有效（准确率提升）
- **Phase 2-3**: OELM在**预训练生成任务**上效果不佳（PPL下降）
- **Phase 5 (当前)**: 验证预训练+微调的完整流程

### 核心假设
| 场景 | 微调 | 预训练 |
|------|------|--------|
| Q/K来源 | 已训练好的表示 | 随机/正交初始化 |
| 冻结效果 | 保持已有知识 | 提供稳定初始化？ |
| 潜在价值 | 参数效率 | 训练稳定性+正则化 |

### 实验配置
| 方法 | 冻结Q/K | 冻结FFN | 可训练参数 | 学习率 |
|------|---------|---------|-----------|--------|
| Baseline | ❌ | ❌ | 100% | 3e-4 |
| OELM-QK | ✅ | ❌ | ~75% | 1e-3 |
| OELM-QK-FFN | ✅ | ✅ | ~65% | 1e-3 |

---

## 2. 目前做到什么程度

### 已完成 ✅

#### 代码编写
- [x] `models/modeling_oelm_pretrain.py` - OELM预训练模型（526行）
- [x] `scripts/train_pretrain.py` - 预训练脚本（已修复DataLoader bug）
- [x] `scripts/finetune_from_pretrain.py` - 下游微调脚本
- [x] `scripts/analyze_results.py` - 结果分析工具

#### 启动脚本
- [x] `run_pretrain_baseline.sh` - Baseline预训练
- [x] `run_pretrain_oelm_qk.sh` - OELM-QK预训练
- [x] `run_pretrain_oelm_qk_ffn.sh` - OELM-QK-FFN预训练
- [x] `run_finetune_*.sh` - 三个微调脚本
- [x] `submit_all_finetune.sh` - 一键提交微调
- [x] `monitor.sh` - 监控工具
- [x] `quick_analyze.sh` - 快速状态检查

#### 文档
- [x] `README.md` - 项目说明
- [x] `EXPERIMENT_TRACKING.md` - 实验跟踪

### 当前状态 ⏳

#### 最新作业
| 作业ID | 方法 | 状态 | 备注 |
|--------|------|------|------|
| **43711** | **Baseline** | **Pending** | 等待GPU资源 |

#### 历史作业
- 43704: Baseline - 因DataLoader bug失败，已修复并重新提交

#### Bug修复记录
**问题**: DataLoader报错 `stack expects each tensor to be equal size`
**原因**: 序列长度不一致，缺少collate_fn
**修复**: 在`train_pretrain.py`中添加了padding collate_fn

---

## 3. 下一步要做什么

### 阶段1: 预训练 (当前)

#### 等待队列
1. **Baseline** (Job 43711) - 等待GPU资源
   ```bash
   # 检查状态
   ssh ntu-cluster 'squeue -u tianyu016'
   
   # 查看日志（启动后）
   ssh ntu-cluster 'tail -f /projects/LlamaFactory/OELM-Pretrain/logs/pretrain_baseline-43711.out'
   ```

2. **OELM-QK** - Baseline完成后提交
   ```bash
   cd /projects/LlamaFactory/OELM-Pretrain
   sbatch scripts/run_pretrain_oelm_qk.sh
   ```

3. **OELM-QK-FFN** - OELM-QK完成后提交
   ```bash
   cd /projects/LlamaFactory/OELM-Pretrain
   sbatch scripts/run_pretrain_oelm_qk_ffn.sh
   ```

#### 预计时间
- 每个预训练作业: 4-8小时（1 epoch TinyStories）
- 总计: ~24小时（顺序运行）

### 阶段2: 下游微调 (预训练完成后)

```bash
# 一键提交所有微调作业
cd /projects/LlamaFactory/OELM-Pretrain
./scripts/submit_all_finetune.sh

# 或单独提交
sbatch scripts/run_finetune_baseline.sh
sbatch scripts/run_finetune_qk.sh
sbatch scripts/run_finetune_qk_ffn.sh
```

### 阶段3: 结果分析

```bash
# 快速状态检查
./scripts/quick_analyze.sh

# 详细分析
python3 scripts/analyze_results.py
```

---

## 4. 关键文件位置

```
/projects/LlamaFactory/OELM-Pretrain/
├── README.md                      # 项目说明
├── EXPERIMENT_TRACKING.md         # 实验跟踪（需更新）
├── models/
│   └── modeling_oelm_pretrain.py  # 模型定义
├── scripts/
│   ├── train_pretrain.py          # 预训练脚本 ✅已修复
│   ├── finetune_from_pretrain.py  # 微调脚本
│   ├── analyze_results.py         # 结果分析
│   ├── monitor.sh                 # 监控工具
│   ├── run_pretrain_*.sh          # 预训练启动脚本
│   └── run_finetune_*.sh          # 微调启动脚本
├── logs/                          # 训练日志
│   └── pretrain_*-{jobid}.out
├── outputs/                       # 输出结果
│   ├── pretrain/                  # 预训练模型
│   │   ├── baseline/
│   │   ├── oelm_qk/
│   │   └── oelm_qk_ffn/
│   └── finetune/                  # 微调结果
└── data/                          # 数据缓存
```

---

## 5. 常用命令速查

### SSH连接
```bash
ssh tianyu016@10.97.216.128
```

### 监控实验
```bash
cd /projects/LlamaFactory/OELM-Pretrain

# 查看作业状态
watch squeue -u tianyu016

# 快速状态检查
./scripts/quick_analyze.sh

# 详细监控
./scripts/monitor.sh

# 查看日志
tail -f logs/pretrain_baseline-*.out
```

### 提交作业
```bash
# 单个预训练
sbatch scripts/run_pretrain_baseline.sh

# 所有微调
./scripts/submit_all_finetune.sh
```

### 取消作业
```bash
scancel <job_id>
```

---

## 6. 成功标准

### 预训练
- **PPL差距**: OELM vs Baseline < 10%
- **训练稳定**: 无NaN/Inf

### 下游微调
- **准确率**: 与Baseline相当或更好（差距 > -2%）
- **参数效率**: 75%/65%可训练参数

---

## 7. 注意事项

### GPU限制
- **配额**: 最多使用2块 Pro 6000 GPU
- **策略**: 一次只能运行1个预训练作业（避免排队）

### 数据缓存
- **位置**: `/projects/LlamaFactory/.cache/huggingface/`
- **首次**: 下载需要较长时间
- **后续**: 会复用缓存

### 常见问题
| 问题 | 解决 |
|------|------|
| QOSMaxGRESPerUser | 等待GPU资源释放 |
| CUDA OOM | 减小batch_size |
| DataLoader错误 | 已修复（collate_fn） |

---

## 8. 联系信息

- **项目负责人**: tianyu016
- **集群地址**: 10.97.216.128
- **项目路径**: `/projects/LlamaFactory/OELM-Pretrain/`

---

**最后更新**: 2026-03-09 08:00 UTC  
**状态**: 等待GPU资源，准备开始Baseline预训练