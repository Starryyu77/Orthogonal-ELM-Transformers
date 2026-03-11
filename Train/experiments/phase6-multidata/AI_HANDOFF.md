# Phase 6 执行上下文记录

**记录时间**: 2026-03-11 14:05  
**执行阶段**: Phase 6 - 多数据集验证  
**状态**: 🟡 脚本已创建，等待手动上传到集群执行

---

## 🎯 当前任务

执行Phase 6多数据集验证实验，在4个数据集上测试OELM方法：
1. AG News (4分类新闻) ☐ ☐ ☐
2. SST-2 (2分类短文本情感) ☐ ☐ ☐
3. XNLI (3分类自然语言推理) ☐ ☐ ☐
4. MNLI (3分类大规模NLI) ☐ ☐ ☐

每个数据集测试3种方法：Baseline / OELM-QK / OELM-QK-FFN

---

## ✅ 已完成工作

| 任务 | 状态 | 详情 |
|:-----|:----:|:-----|
| 实验方案设计 | ✅ | EXPERIMENT_PLAN.md |
| 消融实验方案 | ✅ | ABPLATION_PLAN.md |
| 长期路线图 | ✅ | ROADMAP.md |
| 训练脚本创建 | ✅ | finetune_multidata.py (支持4数据集) |
| AG News脚本 | ✅ | 3个sh文件 |
| SST-2脚本 | ✅ | 3个sh文件 |
| XNLI脚本 | ✅ | 3个sh文件 |
| MNLI脚本 | ✅ | 3个sh文件 |
| 批量提交脚本 | ✅ | submit_all_phase6.sh |
| 执行指南 | ✅ | EXECUTION_GUIDE.md |

---

## 📦 已创建的文件清单

```
Train/experiments/phase6-multidata/
├── README.md                      # 文档导航
├── EXPERIMENT_PLAN.md             # 实验总方案
├── ABPLATION_PLAN.md              # 消融实验方案
├── ROADMAP.md                     # 长期路线图
├── AI_HANDOFF.md                  # 本文件
├── EXECUTION_GUIDE.md             # 执行指南
└── scripts/                       # 执行脚本 (13个文件)
    ├── finetune_multidata.py
    ├── run_agnews_baseline.sh
    ├── run_agnews_qk.sh
    ├── run_agnews_qk_ffn.sh
    ├── run_sst2_baseline.sh
    ├── run_sst2_qk.sh
    ├── run_sst2_qk_ffn.sh
    ├── run_xnli_baseline.sh
    ├── run_xnli_qk.sh
    ├── run_xnli_qk_ffn.sh
    ├── run_mnli_baseline.sh
    ├── run_mnli_qk.sh
    ├── run_mnli_qk_ffn.sh
    └── submit_all_phase6.sh
```

**本地路径**: `/Users/starryyu/2026/Orthogonal ELM Transformers/Train/experiments/phase6-multidata/`

---

## 🚀 下一步操作

由于SSH需要密码认证，请**手动执行**以下步骤：

### 步骤1: 登录集群
```bash
ssh tianyu016@10.97.216.128
cd /projects/LlamaFactory/OELM-Pretrain
```

### 步骤2: 创建目录
```bash
mkdir -p scripts logs outputs/phase6_multidata
```

### 步骤3: 复制文件 (在本地Mac执行)
```bash
cd "/Users/starryyu/2026/Orthogonal ELM Transformers/Train/experiments/phase6-multidata/scripts"
scp *.py tianyu016@10.97.216.128:/projects/LlamaFactory/OELM-Pretrain/scripts/
scp *.sh tianyu016@10.97.216.128:/projects/LlamaFactory/OELM-Pretrain/scripts/
```

### 步骤4: 添加权限并执行 (在集群执行)
```bash
ssh tianyu016@10.97.216.128
cd /projects/LlamaFactory/OELM-Pretrain
chmod +x scripts/*.sh
./scripts/submit_all_phase6.sh
```

---

## 📋 实验配置

### 预训练模型位置
```
/projects/LlamaFactory/OELM-Pretrain/outputs/pretrain/
├── baseline/final_model.pt          (124.4M params)
├── oelm_qk/final_model.pt           (110.2M params, 88.6%)
└── oelm_qk_ffn/final_model.pt       (53.6M params, 43.1%)
```

### 三种对比方法
| 方法 | 学习率 | 冻结设置 | 可训练参数 |
|:-----|:------:|:---------|:----------:|
| Baseline | 3e-4 | 无 | 100% |
| OELM-QK | 1e-3 | Q/K | 88.6% |
| OELM-QK-FFN | 1e-3 | Q/K + FFN | 43.1% |

---

## 🔍 监控命令

```bash
# 查看作业队列
squeue -u tianyu016

# 查看AG News日志
tail -f logs/agnews_baseline_*.out

# 查看结果
cat outputs/phase6_multidata/ag_news/baseline/results.json
```

---

## 📈 预期结果

参考Phase 5 IMDB结果:
- Baseline: 82.95%
- OELM-QK: 84.06% (+1.11%)
- OELM-QK-FFN: 84.78% (+1.83%)

目标: 在4个新数据集上复现类似趋势

---

## 📝 实验进度追踪

| 数据集 | Baseline | OELM-QK | OELM-QK-FFN | 状态 |
|:-------|:--------:|:-------:|:-----------:|:----:|
| **AG News** | ☐ | ☐ | ☐ | 🟡 等待执行 |
| **SST-2** | ☐ | ☐ | ☐ | 🟡 等待执行 |
| **XNLI** | ☐ | ☐ | ☐ | 🟡 等待执行 |
| **MNLI** | ☐ | ☐ | ☐ | 🟡 等待执行 |

---

## ⏱️ 预计时间表

| 数据集 | 预计时间 | 优先级 |
|:-------|:--------:|:------:|
| AG News | ~3小时 | P0 |
| SST-2 | ~1小时 | P0 |
| XNLI | ~6小时 | P1 |
| MNLI | ~6小时 | P1 |
| **总计** | **~16小时** | - |

(使用1块GPU串行执行)

---

## ⚠️ 注意事项

1. **GPU配额**: 一次只能申请1块Pro 6000
2. **存储空间**: 定期检查磁盘空间
3. **作业排队**: 如果作业显示PD状态，表示正在排队等待GPU
4. **日志保存**: 所有输出保存到 `logs/` 和 `outputs/phase6_multidata/`

---

## 🔗 相关文档

| 文档 | 路径 |
|:-----|:-----|
| 执行指南 | [EXECUTION_GUIDE.md](./EXECUTION_GUIDE.md) |
| 实验方案 | [EXPERIMENT_PLAN.md](./EXPERIMENT_PLAN.md) |
| 消融方案 | [ABPLATION_PLAN.md](./ABPLATION_PLAN.md) |
| 路线图 | [ROADMAP.md](./ROADMAP.md) |

---

**上次更新**: 2026-03-11 14:05  
**下次更新**: 任务提交后
