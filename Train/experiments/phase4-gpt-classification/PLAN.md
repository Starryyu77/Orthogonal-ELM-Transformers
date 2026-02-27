# Phase 4: GPT分类任务OELM验证实验 - 实施计划

> 状态: **待确认** | 创建时间: 2026-02-12

---

## 1. 实验背景与目标

### 核心问题
之前实验发现：
- **BERT + 分类任务**: OELM有效 ✅ (+1.08%准确率, +57%速度)
- **GPT + 生成任务**: OELM无效 ❌ (-9.8%~-15.5%性能损失)

**关键疑问**: GPT上OELM失败是因为「解码器架构」还是「生成任务特性」？

### 验证假设
如果**任务类型**决定OELM效果，那么GPT做分类任务也应该有效。

---

## 2. 实验设计

### 2.1 任务选择

| 数据集 | 分类数 | 任务类型 | 数据规模 |
|--------|--------|----------|----------|
| **IMDB** | 2 | 情感分析 | 50K训练 / 25K测试 |
| **AGNews** | 4 | 新闻分类 | 120K训练 / 7.6K测试 |
| **XNLI** | 3 | 自然语言推理 | 392K训练 / 2.5K验证 |
| **MNLI** | 3 | 自然语言推理 | 392K训练 / 20K验证 |

### 2.2 实验列表

| ID | 数据集 | 方法 | GPU | 预计时间 |
|----|--------|------|-----|----------|
| GPT-CLS-01 | IMDB | Baseline | 0 | 4-6小时 |
| GPT-CLS-02 | IMDB | OELM-Freeze | 1 | 4-6小时 |
| GPT-CLS-03 | AGNews | Baseline | 0 | 4-6小时 |
| GPT-CLS-04 | AGNews | OELM-Freeze | 1 | 4-6小时 |
| GPT-CLS-05 | XNLI | Baseline | 0 | 8-12小时 |
| GPT-CLS-06 | XNLI | OELM-Freeze | 1 | 8-12小时 |
| GPT-CLS-07 | MNLI | Baseline | 0 | 8-12小时 |
| GPT-CLS-08 | MNLI | OELM-Freeze | 1 | 8-12小时 |

---

## 3. 技术实现

### 3.1 代码结构

```
experiments/phase4-gpt-classification/
├── README.md                          # 实验说明
├── PLAN.md                            # 本计划
├── REPORT.md                          # 实验报告（完成后）
├── models/
│   ├── __init__.py
│   ├── modeling_gpt_classification.py # Baseline分类模型
│   └── modeling_oelm_classification.py # OELM分类模型
├── scripts/
│   ├── train_classification.py        # 训练脚本
│   ├── run_imdb_experiments.sh        # IMDB实验
│   ├── run_agnews_experiments.sh      # AGNews实验
│   ├── run_xnli_experiments.sh        # XNLI实验
│   ├── run_mnli_experiments.sh        # MNLI实验
│   └── run_all_phase4.sh              # 全部实验
├── data/
│   └── prepare_classification_data.py # 数据准备
└── configs/
    └── experiment_configs.yaml        # 配置
```

### 3.2 模型改造要点

**GPTForSequenceClassification**:
- 移除语言建模头 (`lm_head`)
- 添加分类头: `Linear(d_model, num_classes)`
- 取最后一个有效token的hidden state
- 支持双向attention（非因果）

**OELMForSequenceClassification**:
- 复用phase2的`HeadWiseOrthogonalMultiHeadAttention`
- Q/K分头正交初始化 + 冻结
- V/O和分类头可训练

### 3.3 复用代码

| 来源 | 文件 | 复用内容 |
|------|------|----------|
| phase2-gpt-oelm | `models/modeling_gpt.py` | GPT transformer层 |
| phase2-gpt-oelm | `models/modeling_oelm_v2.py` | 分头正交实现 |
| phase1-bert-xnli | `models/train_bert.py` | 分类训练流程 |

---

## 4. 实施步骤

### Phase 4.1: 基础设施 (Day 1-2)

- [ ] 创建目录结构
- [ ] 实现 `modeling_gpt_classification.py`
- [ ] 实现 `modeling_oelm_classification.py`
- [ ] 实现数据加载脚本

### Phase 4.2: 训练框架 (Day 3-4)

- [ ] 实现 `train_classification.py`
- [ ] 本地测试验证（100 steps快速测试）

### Phase 4.3: 实验脚本 (Day 5)

- [ ] 创建8个实验的启动脚本
- [ ] 编写配置文件

### Phase 4.4: 集群执行 (Day 6-14)

- [ ] 同步代码到NTU集群
- [ ] 执行8个实验（2 GPU并行）
- [ ] 监控训练过程
- [ ] 记录结果

### Phase 4.5: 结果分析 (Day 15)

- [ ] 汇总所有实验结果
- [ ] 撰写REPORT.md
- [ ] 得出结论

---

## 5. 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| GPT分类效果本身不佳 | 中 | 高 | 先跑IMDB验证可行性 |
| XNLI/MNLI句子对处理复杂 | 中 | 中 | 参考phase1实现 |
| 内存不足 | 中 | 高 | 使用gradient accumulation |
| 训练不稳定 | 低 | 中 | warmup + gradient clipping |

---

## 6. 资源需求

### 计算资源
- **GPU**: RTX A5000 (使用2个并行)
- **存储**: ~50GB (模型 + 数据)
- **内存**: ~32GB
- **时间**: ~7-10天

### 单实验配置
```yaml
model:
  d_model: 512
  num_layers: 6
  num_heads: 8
  d_ff: 2048
  max_seq_len: 512

training:
  batch_size: 16
  max_steps: 10000
  warmup_steps: 500
  learning_rate:
    baseline: 3e-4
    oelm_freeze: 1e-3
  gradient_accumulation: 2
```

---

## 7. 预期结果解读

| 场景 | 条件 | 结论 |
|------|------|------|
| **场景A** | GPT-CLS OELM >= Baseline - 2% | **任务类型决定论**: OELM适用于分类任务 |
| **场景B** | GPT-CLS OELM < Baseline - 5% | **架构决定论**: 解码器架构不适合OELM |
| **场景C** | 结果在-2%到-5%之间 | 需进一步分析 |

---

## 8. 决策选项

### 选项1: 试点验证 (推荐)
先只跑IMDB两个实验验证可行性，确认有效后再跑全部。

### 选项2: 全部执行
直接按顺序跑全部8个实验。

### 选项3: 取消
如果认为这个实验方向价值不大。

---

## 9. 执行环境

- **服务器**: 10.97.216.128
- **用户名**: tianyu016
- **项目路径**: `/projects/Orthogonal_ELM_Transformers/Train`
- **执行方式**: tmux会话 + shell脚本

---

**等待确认**: 请回复选择：
1. 试点验证（先跑IMDB）
2. 全部执行
3. 取消实验
4. 修改计划（请说明）
