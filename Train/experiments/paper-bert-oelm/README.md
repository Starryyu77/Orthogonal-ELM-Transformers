# BERT OELM: Head-wise Orthogonal Initialization for Reservoir Computing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **正交极限学习机 Transformer：分头正交初始化与储层计算验证**

本项目验证了**分头正交初始化 (Head-wise Orthogonality)** 在 BERT 模型上的有效性，证明冻结 Query/Key 参数仅训练 87.1% 的参数即可达到全参数微调 98.5% 的性能。

---

## 🎯 核心贡献

1. **分头正交初始化**: 修复了全局正交导致的性能崩塌问题
2. **参数高效训练**: 冻结 12.9% 参数，仅损失 ~1.5% 准确率
3. **跨任务验证**: 在 SST-2 (2分类) 和 MNLI (3分类NLI) 上均有效
4. **正交性必要性**: 消融实验证明正交初始化是必需的

---

## 📊 主要结果

| 数据集 | 任务 | Baseline | OELM-Freeze | 差距 | OELM 达到比例 |
|--------|------|----------|-------------|------|---------------|
| **SST-2** | 2分类情感分析 | 93.12% | 91.28% | -1.84% | 98.0% |
| **MNLI** | 3分类NLI | 83.44% | 82.23% | -1.21% | 98.5% |

**消融实验**:
| 实验 | 准确率 | 结论 |
|------|--------|------|
| OELM-Orthogonal | 91.28% | ✅ 正交初始化有效 |
| OELM-Random | 82.11% | ❌ 随机初始化失败 (-9.17%) |

---

## 📁 项目结构

```
bert-oelm-paper/
├── src/                          # 源代码
│   ├── modeling_bert_oelm.py    # 分头正交初始化实现
│   ├── train_bert.py            # 训练脚本
│   └── __init__.py
├── scripts/                      # 实验脚本
│   ├── run_experiment.sh        # 快速实验启动
│   └── run_fair_comparison.sh   # 公平对比实验
├── configs/                      # 配置文件
│   ├── sst2_baseline.yaml
│   ├── sst2_oelm.yaml
│   ├── mnli_baseline.yaml
│   └── mnli_oelm.yaml
├── experiments/                  # 实验配置
│   ├── sst2/                    # SST-2 实验配置
│   ├── mnli/                    # MNLI 实验配置
│   └── ablation/                # 消融实验配置
├── results/                      # 实验结果
│   ├── sst2/                    # SST-2 训练日志
│   ├── mnli/                    # MNLI 训练日志
│   ├── ablation/                # 消融实验日志
│   └── timing/                  # 计时分析数据
├── figures/                      # 论文图表 (待生成)
├── docs/                         # 文档
│   └── EXPERIMENT_REPORT.md     # 完整实验报告
└── README.md                     # 本文件
```

---

## 🚀 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/bert-oelm.git
cd bert-oelm-paper

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install torch transformers datasets scikit-learn tqdm numpy
```

### 运行实验

#### SST-2 实验

```bash
# Baseline (全参数微调)
python src/train_bert.py \
    --freeze_mode false \
    --lr 2e-5 \
    --dataset sst2 \
    --output_dir outputs/sst2_baseline

# OELM-Freeze (冻结 Q/K)
python src/train_bert.py \
    --freeze_mode true \
    --lr 1e-4 \
    --dataset sst2 \
    --init_method orthogonal \
    --output_dir outputs/sst2_oelm
```

#### MNLI 实验

```bash
# Baseline
python src/train_bert.py \
    --freeze_mode false \
    --lr 2e-5 \
    --dataset mnli \
    --output_dir outputs/mnli_baseline

# OELM-Freeze
python src/train_bert.py \
    --freeze_mode true \
    --lr 1e-4 \
    --dataset mnli \
    --init_method orthogonal \
    --output_dir outputs/mnli_oelm
```

#### 消融实验 (OELM-Random)

```bash
python src/train_bert.py \
    --freeze_mode true \
    --lr 1e-4 \
    --dataset sst2 \
    --init_method normal \
    --output_dir outputs/oelm_random
```

---

## 🔬 核心算法

### 分头正交初始化

```python
def apply_head_wise_orthogonal_(weight: nn.Parameter, num_heads: int) -> None:
    """
    分头正交初始化 - 核心创新

    输入: [hidden_dim, hidden_dim] = [768, 768]
    重塑: [num_heads, head_dim, hidden_dim] = [12, 64, 768]
    处理: 对每个 head 独立 QR 分解
    输出: [hidden_dim, hidden_dim]
    """
    with torch.no_grad():
        hidden_dim = weight.size(0)
        head_dim = hidden_dim // num_heads

        # 重塑为 [num_heads, head_dim, hidden_dim]
        w = weight.view(num_heads, head_dim, hidden_dim).clone()

        # 对每个 head 独立 QR 分解
        for i in range(num_heads):
            q, r = torch.linalg.qr(w[i].T, mode='reduced')
            signs = torch.sign(torch.diag(r))
            q = q * signs.unsqueeze(0)
            w[i] = q.T

        weight.copy_(w.view(hidden_dim, hidden_dim))
```

---

## 📈 实验复现

### 公平对比实验

```bash
# 运行 3 轮 AB-AB 交叉验证
./scripts/run_fair_comparison.sh 3
```

### 关键参数

| 参数 | Baseline | OELM-Freeze | 说明 |
|------|----------|-------------|------|
| 冻结 Q/K | ❌ | ✅ | 核心区别 |
| 学习率 | 2e-5 | 1e-4 | OELM 使用更大学习率 |
| Batch Size | 32 | 32 | 保持一致 |
| Epochs | 3 | 3 | 保持一致 |
| Warmup | 10% | 10% | 保持一致 |

---

## 📚 文档

- [完整实验报告](docs/EXPERIMENT_REPORT_BERT_RESERVOIR.md) - 包含详细方法、结果和讨论
- [训练日志分析](results/) - 所有实验的原始日志
- [计时分析](results/timing/) - 公平对比实验的详细计时数据

---

## 🏆 主要发现

1. **参数效率**: 冻结 12.9% 参数，性能仅下降 1.5%
2. **训练速度**: OELM-Freeze 与 Baseline 无显著差异 (+1.4%)
3. **训练稳定性**: OELM-Freeze 更稳定 (CV 1.0% vs 9.9%)
4. **正交性必要**: OELM-Random 比 OELM-Orthogonal 低 9.17%
5. **泛化能力**: 在复杂 MNLI 任务上差距仅 1.21%

---

## 🔧 环境要求

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Transformers**: 4.x
- **GPU**: NVIDIA GPU with 16GB+ VRAM (推荐 24GB)
- **CUDA**: 11.8+

---

## 📝 引用

如果本工作对您的研究有帮助，请引用：

```bibtex
@article{zhang2025bertoelm,
  title={BERT OELM: Head-wise Orthogonal Initialization for Efficient Transformer Fine-tuning},
  author={Zhang, Tianyu},
  year={2025},
  institution={NTU MLDA Lab}
}
```

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。

---

## 🙏 致谢

- 指导单位: NTU MLDA Lab
- GPU 支持: MLDA GPU Cluster (gpu43.dynip.ntu.edu.sg)
- 代码辅助: Claude Code AI Assistant

---

**最后更新**: 2026-02-08
