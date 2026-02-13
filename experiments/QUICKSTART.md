# 快速开始指南

## 目录

1. [查看实验状态](#查看实验状态)
2. [启动新实验](#启动新实验)
3. [监控运行中实验](#监控运行中实验)
4. [分析结果](#分析结果)

---

## 查看实验状态

### 所有实验概览

```bash
# 查看实验配置文件
cat experiments/phase3-gpt-ablation/configs/experiments.json

# 查看结果汇总
cat experiments/phase3-gpt-ablation/results/summary.md
```

### 已完成实验

| 实验ID | 数据集 | 状态 | PPL | 训练时间 |
|--------|--------|------|-----|----------|
| GPT-01 | TinyStories | ✅ | 4.27 | 4.5h |
| GPT-02 | TinyStories | ✅ | 4.69 | 4.5h |
| GPT-03 | TinyStories | ✅ | 4.97 | 4.25h |
| GPT-04 | OpenWebText | ✅ | 47.24 | 9h |
| GPT-05 | OpenWebText | ✅ | 54.29 | 9h |
| GPT-06 | WikiText-103 | ⏳ | - | - |
| GPT-07 | WikiText-103 | ⏳ | - | - |

---

## 启动新实验

### 方法1: 使用实验专用脚本 (推荐)

```bash
# 进入实验脚本目录
cd experiments/phase3-gpt-ablation/scripts

# 启动单个实验 (指定GPU)
./run_gpt06.sh 2   # WikiText-103 Baseline on GPU 2
./run_gpt07.sh 3   # WikiText-103 OELM-Freeze on GPU 3
```

### 方法2: 使用统一脚本

```bash
cd gpt-oelm-project
./scripts/run_phase3_experiments.sh GPT-06
```

### 方法3: 手动启动

```bash
cd gpt-oelm-project

# WikiText-103 Baseline
CUDA_VISIBLE_DEVICES=2 python3 scripts/train_v2.py \
  --model_type baseline \
  --dataset wikitext103 \
  --max_lr 3e-4 \
  --max_steps 200000 \
  --out_dir outputs/GPT-06_wikitext103_baseline \
  --val_interval 1000 \
  --use_amp
```

---

## 监控运行中实验

### 本地监控

```bash
cd experiments/common/scripts

# 查看状态
./monitor_experiments.sh

# 实时监控 (5秒刷新)
./monitor_experiments.sh live

# 查看特定日志
./monitor_experiments.sh log GPT-06
```

### 远程监控 (MLDA服务器)

```bash
# 查看GPU状态
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg "nvidia-smi"

# 查看训练进度
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg \
  "tmux capture-pane -t gpt06_wikitext103_baseline -p | tail -5"

# 实时查看
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg \
  "tmux attach -t gpt06_wikitext103_baseline"
```

### 查看日志文件

```bash
# 本地查看
tail -f gpt-oelm-project/outputs/GPT-06_wikitext103_baseline/training.log

# 远程查看
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg \
  "tail -f ~/Orthogonal_ELM_Transformers/Train/gpt-oelm-project/outputs/GPT-06_wikitext103_baseline/training.log"
```

---

## 分析结果

### 对比两个实验

```bash
cd experiments/common/scripts

python analyze_results.py \
  --exp1 ../../../gpt-oelm-project/outputs/GPT-04_openwebtext_baseline \
  --exp2 ../../../gpt-oelm-project/outputs/GPT-05_openwebtext_oelm
```

### 查看所有实验汇总

```bash
python analyze_results.py --all
```

### 手动查看结果文件

```bash
# 查看JSON结果
cat gpt-oelm-project/outputs/GPT-04_openwebtext_baseline/timing_stats.json | jq

# 关键字段
# - final_perplexity: 最终困惑度
# - total_formatted: 总训练时间
# - mean_step_time: 平均步时间
```

---

## 常用命令速查

| 操作 | 命令 |
|------|------|
| 查看所有tmux会话 | `tmux ls` |
| 附加到会话 | `tmux attach -t <name>` |
| 分离会话 (不停止) | `Ctrl+B, D` |
| 杀死会话 | `tmux kill-session -t <name>` |
| 查看GPU | `nvidia-smi` |
| 同步代码 | `./mlda-run.sh sync` |
| 查看服务器日志 | `./mlda-run.sh logs` |

---

## 文件导航

```
实验相关:
├── experiments/README.md                      # 主说明
├── experiments/QUICKSTART.md                  # 本文件
├── experiments/phase3-gpt-ablation/PLAN.md    # 实验计划
├── experiments/phase3-gpt-ablation/results/   # 结果汇总
└── experiments/phase3-gpt-ablation/scripts/   # 启动脚本

项目代码:
├── gpt-oelm-project/scripts/train_v2.py       # 训练脚本
├── gpt-oelm-project/models/modeling_oelm.py   # 模型定义
├── gpt-oelm-project/outputs/                  # 实验输出
└── EXPERIMENT_LOG_Phase3.md                   # 详细日志
```

---

*有问题? 查看完整文档: `experiments/README.md` 和 `EXPERIMENT_LOG_Phase3.md`*
