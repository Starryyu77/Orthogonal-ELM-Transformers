# OELM Transformers 本地实验恢复指南

> **说明**: 本文档记录如何在新环境中恢复 OELM (Orthogonal ELM) Transformers 实验项目。

## 项目概述

本项目研究正交ELM (Extreme Learning Machine) Transformers，探索头级正交初始化结合Q/K参数冻结来减少可训练参数的同时保持分类任务性能。

**关键发现**: OELM对分类任务有效（BERT/GPT都适用），但对生成任务无效。

---

## 1. 从GitHub克隆项目

```bash
# 克隆仓库
git clone git@github.com:Starryyu77/Orthogonal-ELM-Transformers.git

# 进入项目目录
cd Orthogonal-ELM-Transformers
```

---

## 2. 环境配置

### 2.1 创建虚拟环境

```bash
# 创建venv
python -m venv venv

# 激活环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2.2 安装依赖

```bash
# 基础依赖
pip install torch>=2.0.0 transformers datasets numpy scikit-learn tqdm

# 或从requirements文件安装（如果存在）
pip install -r requirements.txt
```

---

## 3. 实验目录结构

```
Orthogonal-ELM-Transformers/
├── bert_experiments/          # BERT早期实验
├── experiment_results/        # 本地实验结果（自动生成）
├── Train/
│   ├── experiments/
│   │   ├── paper-bert-oelm/      # BERT论文实验
│   │   ├── phase1-bert-xnli/     # BERT XNLI实验
│   │   ├── phase2-gpt-oelm/      # GPT生成实验
│   │   ├── phase3-gpt-ablation/  # GPT消融实验
│   │   └── phase4-gpt-classification/  # GPT分类实验（OELM有效）
│   └── tools/cluster_setup/      # 集群设置工具
├── README.md
└── LOCAL_SETUP_GUIDE.md         # 本文件
```

---

## 4. 运行实验

### 4.1 BERT + 分类任务（论文实现）

```bash
cd Train/experiments/paper-bert-oelm

# Baseline (全参数微调)
python src/train_bert.py --freeze_mode false --lr 2e-5 --dataset sst2

# OELM-Freeze (冻结Q/K)
python src/train_bert.py --freeze_mode true --lr 1e-4 --dataset sst2 --init_method orthogonal
```

**可用数据集**: sst2, mnli, qnli, qqpar, cola, sts-b, mrpc, rte

### 4.2 GPT + 分类任务（Phase 4）

```bash
cd Train/experiments/phase4-gpt-classification/scripts

# 运行特定实验（指定GPU 0, 1, 2, 3）
./run_imdb_baseline.sh 0
./run_imdb_oelm.sh 1
./run_agnews_baseline.sh 0
./run_agnews_oelm.sh 1
./run_xnli_baseline.sh 0
./run_xnli_oelm.sh 1
./run_mnli_baseline.sh 0
./run_mnli_oelm.sh 1
```

**可用的实验变体**:

| 实验 | 说明 |
|------|------|
| `*_baseline.sh` | 标准微调基准 |
| `*_oelm.sh` | OELM冻结Q/K |
| `*_oelm_ffn_only.sh` | 仅冻结FFN层 |
| `*_oelm_qk_ffn.sh` | 冻结Q/K + FFN |

### 4.3 旧版BERT实验

```bash
cd bert_experiments
python bert_imdb_experiments.py
python bert_agnews_experiments.py
python bert_mnli_experiments.py
python bert_xnli_experiments.py --language en
```

---

## 5. 关键超参数

### OELM-Freeze学习率要求

| 模式 | 学习率 |
|------|--------|
| Baseline | 2e-5 |
| OELM-Freeze | 1e-3 到 1e-4 (3-10× 更高) |

### 参数冻结规则

- **必须冻结**: Query, Key投影
- **绝不能冻结**: Pooler, Classifier, Value, FFN, LayerNorm

---

## 6. 模型文件说明

**注意**: GitHub仓库不包含模型检查点文件(.pt/.pth)，因为它们通常很大(每个几百MB)。

### 本地模型文件位置

运行实验后，模型检查点将保存在:

```
experiment_results/
├── {DATASET}_oelm_ffn_only_lr{LR}/
│   ├── best.pt          # 最佳模型检查点
│   ├── latest.pt        # 最新模型检查点
│   ├── config.json      # 实验配置
│   ├── results.json     # 实验结果
│   └── timing_stats.json # 时间统计
```

### 如果需要恢复训练

只需重新运行相同的实验脚本，训练会重新开始。由于是从预训练模型微调，通常不需要恢复检查点。

---

## 7. NTU EEE GPU集群（远程）

如需在NTU集群上运行:

```bash
# SSH连接
ssh tianyu016@10.97.216.128

# 集群上的项目位置
/projects/Orthogonal_ELM_Transformers/Train

# 使用提供的运行脚本
./mlda-run.sh status          # 检查GPU状态
./mlda-run.sh train-bert-both # 同时运行baseline和OELM
./mlda-run.sh logs-bert       # 查看日志
```

### 本地→集群同步

在本地项目目录中使用Makefile:

```bash
make sync-up      # 本地→集群
make sync-down    # 集群→本地
make submit       # 提交作业
make status       # 查看作业状态
```

---

## 8. 重要文件清单

### 源代码（GitHub上有）

| 文件 | 用途 |
|------|------|
| `Train/experiments/paper-bert-oelm/src/modeling_bert_oelm.py` | BERT OELM模型 |
| `Train/experiments/paper-bert-oelm/src/train_bert.py` | BERT训练脚本 |
| `Train/experiments/phase4-gpt-classification/models/modeling_oelm_classification.py` | GPT OELM分类模型 |
| `Train/experiments/phase4-gpt-classification/scripts/train_classification.py` | GPT分类训练 |

### 实验数据（本地生成）

| 目录 | 内容 |
|------|------|
| `experiment_results/` | 实验结果、模型检查点 |
| `Train/outputs_phase4/` | Phase 4实验输出 |
| `Train/experiments/phase2-gpt-oelm/checkpoints/` | GPT生成实验检查点 |

---

## 9. 继续实验的步骤

### 场景1: 继续已有实验

1. 克隆代码
2. 安装依赖
3. 直接运行相同的脚本，实验会从头开始（数据已保存）

### 场景2: 修改并重新实验

1. 修改模型代码（如 `modeling_oelm_classification.py`）
2. 运行新的实验脚本
3. 结果会保存到新的目录

### 场景3: 分析已有结果

```python
import json

# 读取实验结果
with open('experiment_results/IMDB_oelm_qk_ffn_lr1e-3/results.json') as f:
    results = json.load(f)
    print(f"Best accuracy: {results['best_acc']}")
    print(f"Final accuracy: {results['final_acc']}")
```

---

## 10. 快速参考命令

```bash
# 检查实验状态
ls experiment_results/

# 查看实验结果
cat experiment_results/*/results.json | grep best_acc

# 计算总存储占用
du -sh experiment_results/
du -sh Train/outputs_phase4/

# 清理旧检查点（保留best.pt）
rm experiment_results/*/latest.pt
rm experiment_results/*/checkpoint_*.pt
```

---

## 11. 注意事项

1. **存储空间**: 模型检查点占用大量空间，定期清理不需要的latest.pt和中间检查点
2. **Git管理**: 不要将.pt/.pth文件提交到Git，它们已被.gitignore排除
3. **实验记录**: 每次实验的配置和结果都保存在对应的config.json和results.json中
4. **可复现性**: 随机种子已设置在脚本中，相同配置应产生相似结果

---

## 12. 联系方式

如有问题，请通过GitHub Issues联系。

**仓库地址**: https://github.com/Starryyu77/Orthogonal-ELM-Transformers

---

*最后更新: 2026-03-08*
