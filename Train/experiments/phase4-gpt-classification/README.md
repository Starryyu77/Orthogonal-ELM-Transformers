# Phase 4: GPT分类任务OELM验证实验

## 实验目标

验证OELM在GPT+分类任务上的效果，区分是「解码器架构」还是「生成任务特性」导致GPT+OELM失败。

## 核心假设

| 场景 | 条件 | 结论 |
|------|------|------|
| **场景A** | GPT-CLS OELM >= Baseline - 2% | **任务类型决定论**: OELM适用于分类任务 |
| **场景B** | GPT-CLS OELM < Baseline - 5% | **架构决定论**: 解码器架构不适合OELM |

## 文件结构

```
phase4-gpt-classification/
├── README.md                          # 本文件
├── PLAN.md                            # 详细实施计划
├── models/
│   ├── __init__.py
│   ├── modeling_gpt_classification.py # Baseline分类模型
│   └── modeling_oelm_classification.py # OELM分类模型
├── scripts/
│   ├── train_classification.py        # 训练脚本
│   ├── run_imdb_baseline.sh          # IMDB Baseline实验
│   └── run_imdb_oelm.sh              # IMDB OELM实验
└── data/                              # 数据准备脚本
```

## 快速开始

### 本地测试

```bash
# 测试模型
cd models
python modeling_gpt_classification.py
python modeling_oelm_classification.py
```

### 集群执行

```bash
# 同步代码到集群
make sync-up

# SSH到集群
ssh tianyu016@10.97.216.128

# 执行实验
cd /projects/Orthogonal_ELM_Transformers/Train
tmux new -s imdb_baseline
./experiments/phase4-gpt-classification/scripts/run_imdb_baseline.sh 0

# 另一个窗口执行OELM
tmux new -s imdb_oelm
./experiments/phase4-gpt-classification/scripts/run_imdb_oelm.sh 1
```

## 实验配置

| 配置项 | Baseline | OELM-Freeze |
|--------|----------|-------------|
| 模型 | GPT-Classifier | OELM-Classifier |
| d_model | 512 | 512 |
| num_layers | 6 | 6 |
| num_heads | 8 | 8 |
| 冻结Q/K | ❌ | ✅ 正交初始化 |
| 学习率 | 3e-4 | 1e-3 |
| batch_size | 16 | 16 |
| epochs | 3 | 3 |

## 数据集

- **IMDB**: 50K电影评论，2分类（正面/负面）

## 预期结果

成功标准：OELM-Freeze准确率 >= Baseline - 2%

## 参考

- Phase 2 GPT OELM: `experiments/phase2-gpt-oelm/`
- Phase 1 BERT XNLI: `experiments/phase1-bert-xnli/`
