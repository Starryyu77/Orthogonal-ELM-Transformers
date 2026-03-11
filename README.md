# Orthogonal ELM Transformers (OELM)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Efficient Training of Transformers via Orthogonal Initialization and Q/K Freezing**

## 📋 Overview

This repository contains the implementation and experiments for **Orthogonal ELM (Extreme Learning Machine) Transformers**, a novel approach that combines:
- **Head-wise Orthogonal Initialization** for Query/Key projections
- **Q/K Parameter Freezing** during training
- **Bidirectional Attention** for classification tasks

Our experiments demonstrate that OELM consistently improves classification performance across various datasets while reducing training time.

### 🎯 Key Results

| Dataset | Task | Baseline | OELM | Improvement | Speedup |
|:--------|:-----|:--------:|:----:|:-----------:|:-------:|
| **IMDB** | Sentiment (2-class) | 78.56% | **85.70%** | +7.14% | 3.6% |
| **AG News** | News (4-class) | 87.05% | **92.74%** | +5.69% | 5.7% |
| **XNLI** | NLI (3-class) | 46.39% | **57.99%** | +11.60% | **21%** |
| **MNLI** | NLI (3-class) | 45.18% | **56.78%** | +11.60% | 4.3% |
| **Average** | - | - | - | **+8.99%** | **8.7%** |

**Key Finding**: OELM effectiveness depends on **task type** (classification vs. generation), not architecture type (Encoder vs. Decoder).

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Starryyu77/Orthogonal-ELM-Transformers.git
cd Orthogonal-ELM-Transformers

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch>=2.0.0 transformers datasets numpy scikit-learn tqdm
```

### Run Experiments

#### BERT + XNLI (Paper Implementation)
```bash
cd Train/experiments/paper-bert-oelm
python src/train_bert_xnli.py \
    --model_type baseline \
    --num_epochs 3 \
    --batch_size 16
```

#### GPT + Classification
```bash
cd Train/experiments/phase4-gpt-classification

# IMDB Sentiment
./scripts/run_imdb_baseline.sh 0
./scripts/run_imdb_oelm.sh 1

# AG News
./scripts/run_agnews_baseline.sh 0
./scripts/run_agnews_oelm.sh 1

# XNLI NLI
./scripts/run_xnli_baseline.sh 0
./scripts/run_xnli_oelm.sh 1

# MNLI NLI
./scripts/run_mnli_baseline.sh 0
./scripts/run_mnli_oelm.sh 1
```

---

## 📁 Repository Structure

```
Orthogonal-ELM-Transformers/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
│
├── bert_experiments/                  # Early BERT experiments
│   └── (legacy experiments)
│
├── Train/                             # Main training code
│   ├── README.md                      # Train module documentation
│   │
│   ├── experiments/                   # All experiments
│   │   ├── paper-bert-oelm/          # BERT + XNLI (Paper)
│   │   │   ├── README.md
│   │   │   ├── src/
│   │   │   │   ├── train_bert_xnli.py
│   │   │   │   ├── modeling_bert_oelm.py
│   │   │   │   └── utils.py
│   │   │   ├── docs/
│   │   │   │   └── EXPERIMENT_REPORT_BERT_RESERVOIR.md
│   │   │   └── results/
│   │   │
│   │   ├── phase2-gpt-oelm/          # GPT + TinyStories
│   │   │   └── (generation experiments - OELM not effective)
│   │   │
│   │   ├── phase3-gpt-ablation/      # GPT Ablation Studies
│   │   │   ├── REPORT.md
│   │   │   └── (generation task ablation)
│   │   │
│   │   └── phase4-gpt-classification/ # GPT + Classification ✅
│   │       ├── README.md
│   │       ├── REPORT.md              # Phase 4 Main Report
│   │       ├── REPORT_MNLI.md         # MNLI Detailed Report
│   │       ├── MNLI_EXPERIMENT_PROGRESS.md
│   │       ├── PLAN.md
│   │       ├── models/
│   │       │   ├── modeling_gpt_classification.py
│   │       │   └── modeling_oelm_classification.py
│   │       └── scripts/
│   │           ├── train_classification.py
│   │           ├── run_imdb_*.sh
│   │           ├── run_agnews_*.sh
│   │           ├── run_xnli_*.sh
│   │           └── run_mnli_*.sh
│   │
│   ├── tools/                         # Utility tools
│   └── docs/                          # Documentation
│
└── 参考材料/                           # Reference materials (Chinese)
    └── (related papers and resources)
```

---

## 🔬 Methodology

### OELM-Freeze Architecture

```python
class OELMForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Standard Transformer with orthogonal Q/K init
        self.transformer = TransformerEncoder(
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            qk_init='orthogonal',  # Head-wise QR decomposition
        )

        # 2. Freeze Q/K parameters
        for layer in self.transformer.layers:
            layer.attention.W_q.requires_grad = False
            layer.attention.W_k.requires_grad = False

        # 3. Classification head
        self.classifier = nn.Linear(config.d_model, num_classes)
```

### Key Hyperparameters

| Parameter | Baseline | OELM-Freeze | Note |
|:----------|:--------:|:-----------:|:-----|
| Learning Rate | 3e-4 | **1e-3** | 3-4× higher for OELM |
| Batch Size | 16 | 16 | Same |
| Epochs | 2-3 | 2-3 | Same |
| Warmup Steps | 500 | 500 | Same |
| Weight Decay | 0.01 | 0.01 | Same |
| Q/K Frozen | ❌ No | ✅ Yes | Key difference |

---

## 📊 Experiment Results

### Detailed Results

#### 1. BERT + XNLI (Paper)
- **Baseline**: 85.06% accuracy
- **OELM**: 86.14% accuracy
- **Improvement**: +1.08%

See: [`Train/experiments/paper-bert-oelm/docs/EXPERIMENT_REPORT_BERT_RESERVOIR.md`](Train/experiments/paper-bert-oelm/docs/EXPERIMENT_REPORT_BERT_RESERVOIR.md)

#### 2. GPT + Classification (Phase 4)

##### IMDB (2-class Sentiment)
```
Baseline:  78.56%  |  Loss: 0.639  |  Time: 22m 51s
OELM:      85.70%  |  Loss: 0.405  |  Time: 22m 01s
Improvement: +7.14% accuracy, -37% loss, -3.6% time
```

##### AG News (4-class News)
```
Baseline:  87.05%  |  Loss: 0.512  |  Time: 1h 19m 12s
OELM:      92.74%  |  Loss: 0.266  |  Time: 1h 14m 41s
Improvement: +5.69% accuracy, -48% loss, -5.7% time
```

##### XNLI (3-class NLI)
```
Baseline:  46.39%  |  Loss: 1.035  |  Time: 3h 12m 59s
OELM:      57.99%  |  Loss: 0.902  |  Time: 2h 32m 22s
Improvement: +11.60% accuracy, -13% loss, -21% time
```

##### MNLI (3-class NLI, Large Validation)
```
Baseline:  45.18%  |  Loss: 1.041  |  Time: 2h 38m 29s
OELM:      56.78%  |  Loss: 0.899  |  Time: 2h 31m 36s
Improvement: +11.60% accuracy, -14% loss, -4.3% time
```

See detailed reports:
- [Phase 4 Main Report](Train/experiments/phase4-gpt-classification/REPORT.md)
- [MNLI Detailed Report](Train/experiments/phase4-gpt-classification/REPORT_MNLI.md)
- [OELM-QK-FFN Extension Results](EXPERIMENT_RESULTS_OELM_FFN.md)

---

## 🔑 Key Insights

### 1. Task Type Determines Effectiveness

| Task Type | OELM Effective? | Example |
|:----------|:---------------:|:--------|
| **Classification** | ✅ Yes | IMDB, AG News, XNLI, MNLI |
| **Generation** | ❌ No | TinyStories, OpenWebText |

### 2. Architecture Independence

- **BERT (Encoder)**: ✅ OELM works
- **GPT (Decoder)**: ✅ OELM works (for classification)

### 3. Why NLI Tasks Benefit Most?

1. **Complex Reasoning**: NLI requires understanding relationships between sentences
2. **Bidirectional Context**: Both premise and hypothesis need full context
3. **Orthogonal Q/K**: Maintains diverse attention patterns for nuanced inference

---

## 🛠️ Development

### Cluster Setup (NTU EEE GPU Cluster)

```bash
# SSH to cluster
ssh username@10.97.216.128

# Setup environment
bash Train/tools/cluster_setup/01_ssh_config.sh
bash Train/tools/cluster_setup/02_setup_storage.sh
bash Train/tools/cluster_setup/03_setup_env.sh

# Submit job
sbatch Train/mlda-run.sh
```

### Local Development

```bash
# Install dev dependencies
pip install black flake8 pytest

# Format code
black Train/

# Run tests
pytest Train/tests/
```

---

## 📚 Citation

If you use this code or method in your research, please cite:

```bibtex
@article{oelm2024,
  title={Orthogonal ELM Transformers: Efficient Training via Q/K Freezing},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- NTU EEE GPU Cluster for computational resources
- Hugging Face for Transformers library
- GLUE Benchmark for evaluation datasets

---

## 📞 Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## 🗺️ Roadmap

- [x] BERT + XNLI implementation
- [x] GPT + Classification (IMDB, AG News, XNLI, MNLI)
- [x] OELM-QK-FFN extension (~65% trainable params)
- [ ] Large model validation (GPT-Large, BERT-Large)
- [ ] Long sequence support (1024, 2048)
- [ ] Additional tasks (NER, QA)
- [ ] PyTorch Lightning integration

---

**Last Updated**: 2026-03-08
**Version**: 1.0.0
**Status**: Active Development
