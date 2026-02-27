# MNLI 实验进度记录

## 实验概述
- **实验名称**: GPT-MNLI 分类任务对比实验
- **开始时间**: 2026-02-27 22:36
- **记录时间**: 2026-02-27 22:44
- **预计完成**: 2026-02-28 04:00-05:00 (约5-6小时)

---

## 当前进度状态

### 时间线
| 里程碑 | 时间 | 状态 |
|:-------|:-----|:----:|
| 实验启动 | 22:36 | ✅ 完成 |
| 数据加载 | 22:36-22:38 | ✅ 完成 |
| Epoch 0 开始 | 22:38 | ✅ 运行中 |
| Step 1000 完成 | 22:41 | ✅ 完成 |
| Epoch 0 完成 | ~00:30 | ⏳ 预计 |
| Epoch 1 完成 | ~03:30 | ⏳ 预计 |

### 训练进度

#### Baseline 实验
| 指标 | 数值 | 状态 |
|:-----|:-----|:----:|
| GPU | 0 | 🟢 |
| Epoch | 0/2 | 运行中 |
| Step | 1000/24544 (4.1%) | 🟢 |
| Loss | 1.1140 (↓ from 1.1695) | 🟢 |
| Accuracy | 39.05% | 🟢 |
| Step Time | 190ms | 🟢 |
| GPU利用率 | 99% | 🟢 |
| 显存使用 | 4,712 MiB | 🟢 |
| 温度 | 79°C | 🟡 |

#### OELM-Freeze 实验
| 指标 | 数值 | 状态 |
|:-----|:-----|:----:|
| GPU | 1 | 🟢 |
| Epoch | 0/2 | 运行中 |
| Step | 1000/24544 (4.1%) | 🟢 |
| Loss | 1.1367 (↓ from 1.1853) | 🟢 |
| Accuracy | 36.10% | 🟢 |
| Step Time | 182ms | 🟢 |
| GPU利用率 | 99% | 🟢 |
| 显存使用 | 6,051 MiB | 🟢 |
| 温度 | 78°C | 🟡 |
| 冻结参数 | 3,145,728 (7.0%) | ✅ |

---

## 系统状态

### GPU 整体状态
| GPU | 型号 | 利用率 | 显存使用 | 温度 | 任务 |
|:---:|:-----|:------:|:--------:|:----:|:-----|
| 0 | RTX A5000 | 99% | 4,712 MiB | 79°C | Baseline |
| 1 | RTX A5000 | 99% | 6,051 MiB | 78°C | OELM |
| 2 | RTX A5000 | 0% | 12 MiB | 32°C | 空闲 |
| 3 | RTX A5000 | 0% | 562 MiB | 57°C | 空闲 |

### 进程状态
- **训练进程数**: 6个
- **Baseline PID**: 2051482 (CPU 89%)
- **OELM PID**: 2051596 (CPU 93%)
- **状态**: 全部正常运行

---

## 关键观察

### ✅ 正常指标
1. **GPU满载**: 两个GPU都99%利用率
2. **损失下降**: 两个实验损失都在持续下降
3. **速度正常**: ~190ms/step，符合预期
4. **无错误**: 未发现OOM、Killed或Exception
5. **日志更新**: 每1000 steps正常记录

### ⚠️ 注意事项
1. **温度**: GPU温度79°C/78°C，在合理范围内但需关注
2. **早期准确率**: OELM早期准确率略低于Baseline是正常的（36% vs 39%）
3. **收敛速度**: OELM收敛通常较慢但最终表现更好

---

## 实验配置

### 模型参数
| 参数 | 值 |
|:-----|:---|
| d_model | 512 |
| num_layers | 6 |
| num_heads | 8 |
| d_ff | 2048 |
| max_seq_len | 512 |
| num_classes | 3 |
| 总参数量 | 44,898,307 |

### 训练配置
| 参数 | Baseline | OELM-Freeze |
|:-----|:--------:|:-----------:|
| batch_size | 16 | 16 |
| num_epochs | 2 | 2 |
| learning_rate | 3e-4 | 1e-3 |
| warmup_steps | 500 | 500 |
| weight_decay | 0.01 | 0.01 |
| log_interval | 1000 | 1000 |

### 数据集
- **名称**: MNLI (GLUE)
- **训练集**: 392,702 样本
- **验证集**: 9,815 样本 (matched)
- **分类**: 3类 (entailment, neutral, contradiction)

---

## 日志位置

### 输出目录
```
/usr1/home/s125mdg43_10/Orthogonal_ELM_Transformers/Train/outputs/
├── MNLI_baseline/
│   ├── console.log          # 实时日志
│   ├── config.json          # 实验配置
│   ├── results.json         # 最终结果
│   ├── timing_stats.json    # 时间统计
│   ├── best.pt             # 最佳模型
│   └── latest.pt           # 最终模型
└── MNLI_oelm_freeze/
    ├── console.log
    ├── config.json
    ├── results.json
    ├── timing_stats.json
    ├── best.pt
    └── latest.pt
```

---

## 监控命令

```bash
# 查看实时日志
ssh ntu-gpu43 "tail -f /usr1/home/s125mdg43_10/Orthogonal_ELM_Transformers/Train/outputs/MNLI_baseline/console.log"
ssh ntu-gpu43 "tail -f /usr1/home/s125mdg43_10/Orthogonal_ELM_Transformers/Train/outputs/MNLI_oelm_freeze/console.log"

# 查看GPU状态
ssh ntu-gpu43 "nvidia-smi"

# 查看tmux会话
ssh ntu-gpu43 "tmux ls"
ssh ntu-gpu43 "tmux attach -t mnli_baseline"
ssh ntu-gpu43 "tmux attach -t mnli_oelm"
```

---

## 预期结果

基于XNLI实验结果，预期MNLI：
| 指标 | Baseline | OELM-Freeze | 预期提升 |
|:-----|:--------:|:-----------:|:--------:|
| 验证准确率 | ~50-55% | ~58-63% | +5-8% |
| 训练时间 | ~6h | ~5-5.5h | 快10-15% |

---

## 更新记录

| 时间 | 事件 | 状态 |
|:-----|:-----|:----:|
| 22:36 | 实验启动 | ✅ |
| 22:38 | 数据加载完成，开始训练 | ✅ |
| 22:41 | Step 1000 完成 | ✅ |
| 22:44 | 创建进度记录文档 | ✅ |
| 23:33 | 进度检查 - Epoch 0 约70% | ✅ |

---

**下次检查时间**: Epoch 0 完成后 (~00:00)
**预计Epoch 0完成**: Baseline ~00:05, OELM ~23:55
