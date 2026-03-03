# OELM-FFN 扩展实验

在原有OELM基础上，将FFN（Feed-Forward Network）的升维和降维矩阵也替换为冻结的正交矩阵。

## 实验目标

验证当Q/K **和** FFN都冻结时，模型是否仍能保持可接受的分类性能。

## 架构对比

| 组件 | Baseline | OELM-QK | OELM-QK-FFN | OELM-FFN-only |
|------|----------|---------|-------------|---------------|
| Q/K  | 可训练 | **冻结** | **冻结** | 可训练 |
| V/O  | 可训练 | 可训练 | 可训练 | 可训练 |
| FFN Up (d_model→d_ff) | 可训练 | 可训练 | **冻结** | **冻结** |
| FFN Down (d_ff→d_model) | 可训练 | 可训练 | **冻结** | **冻结** |
| 冻结参数比例 | 0% | ~25% | ~80% | ~64% |

## 核心实现

### FrozenOrthogonalLinear
```python
class FrozenOrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        # 使用torch.nn.init.orthogonal_初始化
        weight = torch.empty(out_features, in_features)
        nn.init.orthogonal_(weight)
        # 注册为buffer（冻结）
        self.register_buffer('weight', weight)
        # 偏置可训练
        self.bias = nn.Parameter(torch.zeros(out_features))
```

### 正交性验证
- 对于方阵：W @ W.T = I
- 对于tall矩阵 (out > in)：W.T @ W = I
- 验证容差：1e-5

## 文件结构

```
models/
├── orthogonal_ffn.py           # 正交FFN模块
├── modeling_oelm_ffn.py        # 完整OELM-FFN模型

tests/
└── test_orthogonal_ffn.py      # 单元测试

scripts/
├── run_imdb_oelm_qk_ffn.sh     # IMDB QK+FFN脚本
├── run_imdb_oelm_ffn_only.sh   # IMDB FFN-only脚本
├── run_agnews_oelm_qk_ffn.sh   # AG News QK+FFN脚本
├── run_agnews_oelm_ffn_only.sh # AG News FFN-only脚本
├── run_xnli_oelm_qk_ffn.sh     # XNLI QK+FFN脚本
├── run_xnli_oelm_ffn_only.sh   # XNLI FFN-only脚本
├── run_mnli_oelm_qk_ffn.sh     # MNLI QK+FFN脚本
└── run_mnli_oelm_ffn_only.sh   # MNLI FFN-only脚本
```

## 使用方法

### 支持的新model_type

```bash
# OELM-QK-FFN (Q/K和FFN都冻结)
python scripts/train_classification.py \
    --model_type oelm_qk_ffn \
    --dataset imdb \
    --learning_rate 1e-3

# OELM-FFN-only (只冻结FFN)
python scripts/train_classification.py \
    --model_type oelm_ffn_only \
    --dataset imdb \
    --learning_rate 5e-4

# 随机初始化版本（消融实验）
python scripts/train_classification.py \
    --model_type oelm_qk_ffn_random \
    --dataset imdb
```

### 使用启动脚本

```bash
cd scripts/

# OELM-QK-FFN (3个学习率)
./run_imdb_oelm_qk_ffn.sh 0 0  # GPU 0, lr=1e-3
./run_imdb_oelm_qk_ffn.sh 1 1  # GPU 1, lr=3e-3
./run_imdb_oelm_qk_ffn.sh 2 2  # GPU 2, lr=1e-2

# OELM-FFN-only (3个学习率)
./run_imdb_oelm_ffn_only.sh 0 0  # GPU 0, lr=5e-4
./run_imdb_oelm_ffn_only.sh 1 1  # GPU 1, lr=1e-3
./run_imdb_oelm_ffn_only.sh 2 2  # GPU 2, lr=3e-3
```

## 学习率建议

由于可训练参数大幅减少，需要调整学习率：

| 实验 | 推荐学习率 | 原因 |
|------|-----------|------|
| OELM-QK-FFN | 1e-3 ~ 1e-2 | 只有20%参数可训练，需要大LR |
| OELM-FFN-only | 5e-4 ~ 3e-3 | 36%参数可训练，中等LR |

## 单元测试

```bash
# 运行所有测试
cd tests/
pytest test_orthogonal_ffn.py -v

# 关键测试点
- test_orthogonality_square: 验证方阵正交性
- test_orthogonality_tall: 验证tall矩阵正交性
- test_frozen_weights: 验证权重被冻结
- test_trainable_bias: 验证偏置可训练
```

## 预期结果

基于OELM-QK的经验，预期：

| 数据集 | Baseline | OELM-QK | OELM-QK-FFN | OELM-FFN-only |
|--------|----------|---------|-------------|---------------|
| IMDB | ~79% | ~86% | ? | ? |
| AG News | ~87% | ~93% | ? | ? |
| XNLI | ~46% | ~58% | ? | ? |

**假设**: 冻结FFN可能导致更大性能下降，因为FFN占模型参数的大部分。

## 实验结果记录

（待运行后填写）

| 实验 | 学习率 | 准确率 | 训练时间 | 备注 |
|------|--------|--------|----------|------|
| IMDB OELM-QK-FFN (lr=1e-3) | | | | |
| IMDB OELM-QK-FFN (lr=3e-3) | | | | |
| IMDB OELM-QK-FFN (lr=1e-2) | | | | |
| IMDB OELM-FFN-only (lr=5e-4) | | | | |
| IMDB OELM-FFN-only (lr=1e-3) | | | | |
| IMDB OELM-FFN-only (lr=3e-3) | | | | |

## 关键发现

（待运行后填写）

---

**创建时间**: 2026-03-03
**状态**: 实施完成，等待实验运行
