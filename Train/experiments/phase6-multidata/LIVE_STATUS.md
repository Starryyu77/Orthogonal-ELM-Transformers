# Phase 6 实验执行状态 - 实时更新

**更新时间**: 2026-03-11 15:07 (Singapore Time)  
**状态**: 🟢 **实验中 - 训练正常进行**

---

## 📊 当前状态

### 运行中的作业

| 作业ID | 数据集 | 方法 | GPU | 运行时间 | 进度 |
|:-------|:-------|:-----|:----|:---------|:-----|
| 44465 | AG News | Baseline | 6000 ADA-2 | 19分30秒 | **Epoch 1/3: 41%** |
| 44466 | AG News | OELM-QK | 6000 ADA-3 | 18分46秒 | 运行中 |

### 排队的作业 (10个)

| 作业ID范围 | 数据集 | 数量 |
|:-----------|:-------|:----:|
| 44467 | AG News OELM-QK-FFN | 1 |
| 44473-44475 | SST-2 | 3 |
| 44476-44478 | XNLI | 3 |
| 44479-44481 | MNLI | 3 |

---

## ✅ 已完成的配置

- ✅ 所有脚本使用 **6000 ADA GPU** (一致性和性能)
- ✅ 修复环境配置 (移除 `module load`, 直接使用 python3)
- ✅ 12个作业全部提交到队列
- ✅ 自动监控已配置 (每15分钟检查完成情况)
- ✅ 结果将自动收集

---

## ⏱️ 预计时间

| 数据集 | 作业数 | 每个预计时间 | 总计 |
|:-------|:------:|:------------:|:----:|
| AG News | 3 | ~3小时 | ~9小时 |
| SST-2 | 3 | ~1小时 | ~3小时 |
| XNLI | 3 | ~6小时 | ~18小时 |
| MNLI | 3 | ~6小时 | ~18小时 |
| **总计** | **12** | - | **~48小时** |

**开始时间**: 2026-03-11 06:47 UTC  
**预计完成**: 2026-03-13 06:47 UTC (约2天后)

---

## 🤖 自动化流程

系统已配置为完全自动化：

```
作业运行 (6000 ADA GPU)
    ↓
每15分钟检查完成状态
    ↓
12/12完成后
    ↓
自动运行 collect_results.py
    ↓
生成 phase6_results_summary.md
    ↓
更新 REPORT.md
    ↓
发送完成通知
```

**无需人工干预！**

---

## 🔍 监控方法

### 方法1: 实时查看作业状态
```bash
ssh ntu-cluster
squeue -u tianyu016
```

### 方法2: 查看训练进度
```bash
ssh ntu-cluster
# AG News Baseline进度
tail -20 scripts/logs/agnews-baseline-44465.err

# AG News OELM-QK进度
tail -20 scripts/logs/agnews-oelm-qk-44466.err
```

### 方法3: 查看完成情况
```bash
ssh ntu-cluster
cat logs/completion_monitor.log
```

### 方法4: 查看结果汇总 (完成后)
```bash
ssh ntu-cluster
cat phase6_results_summary.md
```

---

## 📁 重要文件位置

```
/projects/LlamaFactory/OELM-Pretrain/
├── scripts/logs/                    # 训练日志
│   ├── agnews-baseline-44465.out
│   ├── agnews-baseline-44465.err   # 训练进度
│   └── ...
├── outputs/phase6_multidata/        # 实验结果
│   ├── ag_news/
│   │   ├── baseline/results.json   # (完成后生成)
│   │   ├── oelm_qk/results.json
│   │   └── oelm_qk_ffn/results.json
│   ├── sst2/
│   ├── xnli/
│   └── mnli/
├── logs/completion_monitor.log      # 自动监控日志
├── phase6_results_summary.md        # (完成后生成)
└── check_completion.sh              # 自动监控脚本
```

---

## 📝 历史记录

| 时间 (SGT) | 事件 |
|:-----------|:-----|
| 14:47 | AG News Baseline开始运行 |
| 14:47 | AG News OELM-QK开始运行 |
| 15:07 | Epoch 1/3达到41% |
| 待定 | AG News Baseline完成 |
| 待定 | AG News OELM-QK完成 |
| 待定 | AG News OELM-QK-FFN开始 |
| ... | ... |

---

## 🎯 预期结果

基于Phase 5 (IMDB)结果，预期在4个新数据集上：

| 方法 | 预期准确率 | 相对提升 |
|:-----|:----------:|:--------:|
| Baseline | ~83% | - |
| OELM-QK | ~84% | +1% |
| OELM-QK-FFN | ~85% | +2% |

---

## ⚠️ 注意事项

1. **训练时间较长**: 每个作业1-6小时，总计约48小时
2. **串行执行**: 所有作业在1块6000 ADA GPU上串行执行，确保计时一致性
3. **自动完成**: 系统会自动收集结果，无需持续监控
4. **中断恢复**: 如意外中断，可重新提交未完成的作业

---

**状态**: 🟢 正常运行中  
**下次自动检查**: 每15分钟  
**预计完成**: 2026-03-13
