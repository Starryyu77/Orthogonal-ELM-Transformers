# OELM Transformers 实验完整归档

> 创建时间: 2026-02-11
> 最后更新: 2026-02-11

---

## 归档内容

本次归档包含所有实验的完整记录，包括：

1. ✅ 所有实验数据已同步 (5.0GB)
2. ✅ 实验脚本已整理分类
3. ✅ 说明文档和注释已创建
4. ✅ 实验计划和结果已汇总

---

## 实验概览

### Phase 1: XNLI (已完成)

| 实验 | 模型 | 方法 | 准确率 | 状态 |
|------|------|------|--------|------|
| XNLI-01 | BERT-Base | Baseline | 76.71% | ✅ |
| XNLI-02 | BERT-Base | OELM-Freeze | **77.79%** | ✅ |

**结论**: OELM 在分类任务上有效 (+1.08%)

### Phase 3: GPT 消融 (已完成5/7)

| 实验 | 数据集 | 方法 | PPL | 差距 | 状态 |
|------|--------|------|-----|------|------|
| GPT-01 | TinyStories | Baseline | 4.27 | - | ✅ |
| GPT-02 | TinyStories | OELM-Freeze | 4.69 | +9.8% | ✅ |
| GPT-03 | TinyStories | OELM-Random | 4.97 | +16.4% | ✅ |
| GPT-04 | OpenWebText | Baseline | 47.24 | - | ✅ |
| GPT-05 | OpenWebText | OELM-Freeze | 54.29 | +14.9% | ✅ |
| GPT-06 | WikiText-103 | Baseline | TBD | - | ⏳ |
| GPT-07 | WikiText-103 | OELM-Freeze | TBD | TBD | ⏳ |

**结论**: GPT 上冻结 Q/K 代价大，目标未达成

---

## 文件结构

```
experiments/                              # 实验脚本和文档 (80KB)
├── README.md                            # 主说明文档
├── QUICKSTART.md                        # 快速开始指南
├── EXPERIMENTS_COMPLETE.md              # 本文件
├── phase1-xnli/
│   └── README.md                        # Phase 1 说明
├── phase2-gpt-port/
│   └── README.md                        # Phase 2 说明
├── phase3-gpt-ablation/
│   ├── PLAN.md                          # 实验计划
│   ├── scripts/                         # 启动脚本
│   │   ├── run_gpt01.sh                 # TinyStories Baseline
│   │   ├── run_gpt02.sh                 # TinyStories OELM-Freeze
│   │   ├── run_gpt03.sh                 # TinyStories OELM-Random
│   │   ├── run_gpt04.sh                 # OpenWebText Baseline
│   │   ├── run_gpt05.sh                 # OpenWebText OELM-Freeze
│   │   ├── run_gpt06.sh                 # WikiText-103 Baseline
│   │   └── run_gpt07.sh                 # WikiText-103 OELM-Freeze
│   ├── configs/                         # 配置文件
│   │   ├── datasets.yaml                # 数据集配置
│   │   └── experiments.json             # 实验定义
│   └── results/
│       └── summary.md                   # 结果汇总
└── common/
    └── scripts/
        ├── monitor_experiments.sh       # 监控脚本
        └── analyze_results.py           # 分析脚本

gpt-oelm-project/outputs/                 # 实验数据 (5.0GB)
├── GPT-01_baseline/                     # (514MB × 2)
│   ├── timing_stats.json                # 计时统计
│   ├── training.log                     # 训练日志
│   ├── best.pt                          # 最佳模型
│   └── latest.pt                        # 最新模型
├── GPT-02_oelm_freeze/                  # (490MB × 2)
│   └── ...
├── GPT-03_oelm_random/                  # (490MB × 2)
│   └── ...
├── GPT-04_openwebtext_baseline/         # (514MB × 2)
│   └── ...
└── GPT-05_openwebtext_oelm/             # (490MB × 2)
    └── ...
```

---

## 关键数据文件

| 文件 | 大小 | 内容 |
|------|------|------|
| `timing_stats.json` | ~850B | 完整计时统计 |
| `training.log` | ~10MB | 训练日志 |
| `best.pt` | ~515MB | 最佳模型权重 |
| `latest.pt` | ~515MB | 最新模型权重 |

**总计**: 5.0GB 实验数据

---

## 快速访问

### 查看结果
```bash
# 结果汇总
cat experiments/phase3-gpt-ablation/results/summary.md

# 实验计划
cat experiments/phase3-gpt-ablation/PLAN.md

# 快速开始
cat experiments/QUICKSTART.md
```

### 启动新实验
```bash
cd experiments/phase3-gpt-ablation/scripts
./run_gpt06.sh 2   # WikiText-103 Baseline on GPU 2
./run_gpt07.sh 3   # WikiText-103 OELM-Freeze on GPU 3
```

### 分析结果
```bash
cd experiments/common/scripts
python analyze_results.py --all
```

---

## 关键发现汇总

### 正交初始化效果
- ✅ **有效**: OELM-Freeze 比 OELM-Random 好 6.0%
- ✅ **BERT上有效**: XNLI 上 +1.08%

### 冻结 Q/K 的代价
- ❌ **GPT上代价大**: -9.8% ~ -14.9%
- ❌ **大数据集更差**: OpenWebText (-14.9%) > TinyStories (-9.8%)
- ❌ **无速度优势**: 步时间与 Baseline 相同

### 任务类型差异
| 架构 | 任务 | OELM效果 |
|------|------|----------|
| BERT | 分类 | ✅ 优于 Baseline |
| GPT | 生成 | ❌ 劣于 Baseline |

---

## 后续计划

1. **完成剩余实验**: GPT-06/07 (WikiText-103)
2. **考虑改进策略**:
   - 部分解冻 (只冻结部分层的Q/K)
   - 渐进式解冻
   - 分层学习率
3. **撰写完整报告**

---

## 相关文档

| 文档 | 路径 |
|------|------|
| 详细实验日志 | `EXPERIMENT_LOG_Phase3.md` |
| 主说明文档 | `experiments/README.md` |
| 快速开始 | `experiments/QUICKSTART.md` |
| 实验计划 | `experiments/phase3-gpt-ablation/PLAN.md` |
| 结果汇总 | `experiments/phase3-gpt-ablation/results/summary.md` |

---

*归档完成*
