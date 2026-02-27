# 项目结构说明

本文档详细说明 `bert-oelm-paper` 项目的目录结构和各文件用途。

---

## 目录树

```
bert-oelm-paper/
├── src/                              # 核心源代码
│   ├── __init__.py                   # 模块初始化
│   ├── modeling_bert_oelm.py        # 分头正交初始化实现 (核心)
│   └── train_bert.py                # 训练脚本
│
├── scripts/                          # 实验脚本
│   ├── run_experiment.sh            # 快速实验启动
│   └── run_fair_comparison.sh       # 公平对比实验 (AB-AB模式)
│
├── configs/                          # 实验配置
│   ├── sst2_baseline.yaml           # SST-2 Baseline配置
│   ├── sst2_oelm.yaml               # SST-2 OELM配置
│   ├── mnli_baseline.yaml           # MNLI Baseline配置
│   └── mnli_oelm.yaml               # MNLI OELM配置
│
├── experiments/                      # 实验配置目录 (预留)
│   ├── sst2/                        # SST-2实验相关
│   ├── mnli/                        # MNLI实验相关
│   └── ablation/                    # 消融实验相关
│
├── results/                          # 实验结果与日志
│   ├── sst2/                        # SST-2训练日志
│   │   ├── bert_baseline.log       # Baseline训练日志 (~1.3MB)
│   │   └── bert_oelm.log           # OELM训练日志 (~1.3MB)
│   ├── mnli/                        # MNLI训练日志
│   │   ├── mnli_baseline.log       # Baseline训练日志 (~9.1MB)
│   │   └── mnli_oelm.log           # OELM训练日志 (~9.2MB)
│   ├── ablation/                    # 消融实验日志
│   │   └── oelm_random_ablation.log # OELM-Random日志 (~1.3MB)
│   └── timing/                      # 计时分析数据
│       ├── baseline_run1_*.json    # Baseline计时数据
│       ├── oelm_run1_*.json        # OELM计时数据
│       └── comparison_summary_*.txt # 对比实验摘要
│
├── figures/                          # 论文图表 (待生成)
│   # 存放 Matplotlib/Seaborn生成的图表
│   # - sst2_accuracy_curve.png
│   # - mnli_accuracy_curve.png
│   # - comparison_bar_chart.png
│   # - timing_comparison.png
│
├── data/                             # 数据目录 (预留)
│   # 数据集说明文件
│   # 不存放实际数据文件 (通过HuggingFace下载)
│
├── docs/                             # 文档
│   └── EXPERIMENT_REPORT_BERT_RESERVOIR.md  # 完整实验报告 (~800行)
│
├── README.md                         # 项目README (主要入口)
├── EXPERIMENT_SUMMARY.md            # 实验总结 (快速参考)
├── PROJECT_STRUCTURE.md             # 本文件
├── GITHUB_UPLOAD_GUIDE.md           # GitHub上传指南
│
├── requirements.txt                  # Python依赖
├── LICENSE                           # MIT许可证
├── CITATION.cff                     # 引用格式文件
└── .gitignore                       # Git忽略文件

```

---

## 文件详细说明

### 核心代码 (`src/`)

| 文件 | 行数 | 说明 | 关键函数 |
|------|------|------|----------|
| `modeling_bert_oelm.py` | ~300 | 分头正交初始化实现 | `apply_head_wise_orthogonal_()`, `check_orthogonality()`, `freeze_model_parameters()` |
| `train_bert.py` | ~700 | 训练脚本 | `train()`, `evaluate()`, `load_sst2_data()`, `load_mnli_data()` |

### 实验日志 (`results/`)

| 文件 | 大小 | 内容 | 关键信息 |
|------|------|------|----------|
| `bert_baseline.log` | ~1.3MB | SST-2 Baseline训练 | Val Acc: 93.12%, 6315 steps |
| `bert_oelm.log` | ~1.3MB | SST-2 OELM训练 | Val Acc: 91.28%, 6315 steps |
| `oelm_random_ablation.log` | ~1.3MB | 消融实验 | Val Acc: 82.11%, 验证正交性 |
| `mnli_baseline.log` | ~9.1MB | MNLI Baseline训练 | Val Acc: 83.44%, 36K steps |
| `mnli_oelm.log` | ~9.2MB | MNLI OELM训练 | Val Acc: 82.23%, 36K steps |
| `*.json` | ~150KB | 计时数据 | 每步时间、标准差 |

### 配置文件 (`configs/`)

每个YAML文件包含：
- 实验元数据 (名称、任务、数据集)
- 模型配置 (BERT-base参数)
- 训练配置 (学习率、batch size等)
- 实验结果 (最佳准确率、训练时间等)

---

## 关键路径速查

### 快速开始
```
README.md → src/train_bert.py → results/
```

### 理解方法
```
README.md → src/modeling_bert_oelm.py (核心算法)
```

### 查看结果
```
EXPERIMENT_SUMMARY.md → results/ → configs/
```

### 论文写作
```
docs/EXPERIMENT_REPORT_BERT_RESERVOIR.md → results/ → figures/ (待生成)
```

---

## 文件大小统计

| 目录 | 大小 | 说明 |
|------|------|------|
| `src/` | ~44KB | 源代码 |
| `results/` | ~21MB | 训练日志 (主要空间占用) |
| `docs/` | ~36KB | 文档 |
| `configs/` | ~16KB | 配置文件 |
| `scripts/` | ~20KB | 脚本 |
| **总计** | **~21MB** | (不含figures和data) |

---

## 后续添加文件建议

### 论文写作阶段
```
figures/
├── sst2_training_curve.png
├── mnli_training_curve.png
├── accuracy_comparison.png
├── parameter_efficiency.png
└── timing_comparison.png

paper/
├── main.tex              # LaTeX主文件
├── introduction.tex
├── methodology.tex
├── experiments.tex
├── results.tex
├── conclusion.tex
├── references.bib
└── supplementary.pdf
```

### 代码扩展阶段
```
src/
├── models/
│   ├── __init__.py
│   ├── modeling_bert_oelm.py
│   ├── modeling_roberta_oelm.py  # 扩展到RoBERTa
│   └── modeling_gpt_oelm.py      # 扩展到GPT
├── trainers/
│   ├── __init__.py
│   ├── bert_trainer.py
│   └── base_trainer.py
└── utils/
    ├── __init__.py
    ├── orthogonality.py
    └── visualization.py          # 绘图工具

tests/
├── test_orthogonality.py
├── test_model_loading.py
└── test_training.py
```

---

## GitHub 仓库结构建议

上传后，GitHub 仓库应显示：

```
📦 bert-oelm
├── 📁 src/                 # 代码
├── 📁 scripts/             # 脚本
├── 📁 configs/             # 配置
├── 📁 results/             # 结果
├── 📁 docs/                # 文档
├── 📄 README.md            # 主页显示
├── 📄 EXPERIMENT_SUMMARY.md # 实验总结
├── 📄 requirements.txt     # 依赖
├── 📄 LICENSE              # 许可证
└── 📄 CITATION.cff        # 引用信息
```

---

## 使用建议

1. **复现实验**: 从 `README.md` 开始，按快速开始步骤操作
2. **理解方法**: 阅读 `src/modeling_bert_oelm.py` 的核心函数
3. **查看结果**: 查阅 `EXPERIMENT_SUMMARY.md` 和 `results/` 日志
4. **论文写作**: 基于 `docs/EXPERIMENT_REPORT_BERT_RESERVOIR.md` 扩展
5. **上传GitHub**: 参考 `GITHUB_UPLOAD_GUIDE.md`

---

**最后更新**: 2026-02-08
