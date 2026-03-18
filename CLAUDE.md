# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for **Orthogonal ELM (Extreme Learning Machine) Transformers**, exploring head-wise orthogonal initialization combined with Q/K parameter freezing to reduce trainable parameters while maintaining performance on classification tasks.

**Key Finding**: OELM is effective for classification tasks (BERT/GPT both work) but NOT for generation tasks. Effectiveness depends on task type, not architecture type.

**Recent Update**: OELM-QK-FFN (冻结Q/K+FFN，~65%可训练参数) 在多个数据集上达到与原始OELM相当或更好的性能。

## Common Commands

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch>=2.0.0 transformers datasets numpy scikit-learn tqdm
```

### Running Experiments

**BERT + Classification (Paper Implementation)**:
```bash
cd Train/experiments/paper-bert-oelm

# Baseline (full fine-tuning)
python src/train_bert.py --freeze_mode false --lr 2e-5 --dataset sst2

# OELM-Freeze (frozen Q/K)
python src/train_bert.py --freeze_mode true --lr 1e-4 --dataset sst2 --init_method orthogonal
```

**GPT + Classification (Phase 4)**:
```bash
cd Train/experiments/phase4-gpt-classification/scripts

# Run with specific GPU (0, 1, 2, 3)
./run_imdb_baseline.sh 0
./run_imdb_oelm.sh 1
./run_agnews_baseline.sh 0
./run_agnews_oelm.sh 1
./run_xnli_baseline.sh 0
./run_xnli_oelm.sh 1
./run_mnli_baseline.sh 0
./run_mnli_oelm.sh 1
```

**Legacy BERT Experiments**:
```bash
cd bert_experiments
python bert_imdb_experiments.py
python bert_agnews_experiments.py
python bert_mnli_experiments.py
python bert_xnli_experiments.py --language en
```

**OELM-QK-FFN 扩展实验** (冻结Q/K+FFN，~65%可训练参数):
```bash
cd Train/experiments/phase4-gpt-classification/scripts

# IMDB
./run_imdb_oelm_qk_ffn.sh 0

# AG News
./run_agnews_oelm_qk_ffn.sh 0

# XNLI
./run_xnli_oelm_qk_ffn.sh 0

# MNLI
./run_mnli_oelm_qk_ffn.sh 0
```

**数据准备** (用于GPT生成实验):
```bash
cd Train/tools/data_preparation

# 下载并处理TinyStories
python download_tinystories.py --output_dir ../../data/tinystories

# 下载并处理OpenWebText样本
python download_openwebtext.py --output_dir ../../data/openwebtext --max_samples 100000
```

### Distributed Training (Dual GPU)

```bash
cd Train/experiments/paper-bert-oelm

# DDP training
torchrun --nproc_per_node=2 src/train_bert.py --freeze_mode true --lr 1e-4
```

### NTU EEE GPU Cluster (Remote)

```bash
# SSH to cluster
ssh tianyu016@10.97.216.128

# Project location on cluster
/projects/Orthogonal_ELM_Transformers/Train

# Use the provided run script
./mlda-run.sh status          # Check GPU status
./mlda-run.sh train-bert-both # Run both baseline and OELM
./mlda-run.sh logs-bert       # View logs
```

## High-Level Architecture

### Core Innovation: Head-Wise Orthogonal Initialization

The key architectural component is `HeadWiseOrthogonalLinear` which applies QR decomposition independently per attention head rather than globally:

```
Global QR (WRONG):    [768, 768] -> QR -> destroys per-head structure
Head-wise QR (CORRECT): [12, 64, 768] -> QR per head -> preserves head diversity
```

### Key Components

**1. HeadWiseOrthogonalLinear** (`modeling_oelm_classification.py`, `modeling_bert_oelm.py`)
- Custom Linear layer with head-wise orthogonal initialization
- Optionally freezes weights (for Q/K projections)
- Uses `torch.linalg.qr()` per head with sign correction

**2. HeadWiseOrthogonalMultiHeadAttention**
- Uses `HeadWiseOrthogonalLinear` for Q and K projections
- Standard trainable Linear for V projection
- Supports both causal (generation) and bidirectional (classification) attention

**3. OELMTransformerLayer**
- Pre-norm architecture (LayerNorm before attention/FFN)
- Uses OELM attention for classification tasks

### Model Variants

| Variant | Location | Use Case | Trainable Params |
|---------|----------|----------|------------------|
| BERT OELM | `paper-bert-oelm/src/modeling_bert_oelm.py` | BERT fine-tuning with frozen Q/K | ~75% |
| GPT OELM | `phase4-gpt-classification/models/modeling_oelm_classification.py` | GPT for classification | ~75% |
| GPT Baseline | `phase4-gpt-classification/models/modeling_gpt_classification.py` | Standard GPT classifier | 100% |
| **GPT OELM-QK-FFN** | `phase4-gpt-classification/models/modeling_oelm_classification.py` | **冻结Q/K+FFN** | **~65%** |

### Experiment Organization

```
Train/experiments/
├── paper-bert-oelm/        # Paper experiments: BERT + SST-2/MNLI
├── phase1-bert-xnli/       # Early BERT XNLI experiments
├── phase2-gpt-oelm/        # GPT generation experiments (OELM NOT effective)
├── phase3-gpt-ablation/    # Ablation studies on generation
├── phase4-gpt-classification/  # GPT classification (OELM effective!)
│   ├── models/             # OELM classification models
│   ├── scripts/            # Experiment scripts (baseline, oelm, oelm_qk_ffn)
│   └── data/               # Prepared datasets
└── common/                 # Shared utilities
```

### Critical Hyperparameters

**Learning Rate by Configuration**:

| Configuration | Learning Rate | Trainable Params |
|--------------|---------------|------------------|
| Baseline | 2e-5 | 100% |
| OELM-Freeze (Q/K) | 1e-3 to 1e-4 | ~75% |
| **OELM-QK-FFN** | **1e-3 (推荐)** | **~65%** |

**Dataset-Specific Best LR for OELM-QK-FFN**:
- IMDB: 3e-3
- AG News: 1e-3
- XNLI: 1e-3
- MNLI: 1e-3

**Parameter Freezing Rule**:
- MUST freeze: Query, Key projections (OELM基础)
- OPTIONAL freeze: FFN Up/Down matrices (OELM-QK-FFN扩展)
- MUST NOT freeze: Pooler, Classifier, Value, LayerNorm

## Key File Locations

| Purpose | Path |
|---------|------|
| BERT OELM Model | `Train/experiments/paper-bert-oelm/src/modeling_bert_oelm.py` |
| BERT Training | `Train/experiments/paper-bert-oelm/src/train_bert.py` |
| GPT OELM Model | `Train/experiments/phase4-gpt-classification/models/modeling_oelm_classification.py` |
| GPT Training | `Train/experiments/phase4-gpt-classification/scripts/train_classification.py` |
| Legacy BERT | `bert_experiments/bert_*_experiments.py` |
| Cluster Utils | `Train/tools/cluster_setup/` |
| Data Preparation | `Train/tools/data_preparation/` |
| Local Setup Guide | `LOCAL_SETUP_GUIDE.md` |
| OELM-FFN Results | `EXPERIMENT_RESULTS_OELM_FFN.md` |

## Data Flow

1. Load dataset via HuggingFace `datasets` library
2. Tokenize with model-specific tokenizer (BERT: BertTokenizer, GPT: GPT2Tokenizer)
3. Model applies head-wise orthogonal init to Q/K weights
4. Freeze Q/K parameters (if OELM mode)
5. Train with standard classification objective
6. Evaluate accuracy on validation set

## Important Implementation Details

- **Orthogonality Check**: Always verify `W @ W^T ≈ I` after initialization using `check_orthogonality()`
- **Pooler Integrity**: Pooler must remain trainable (critical for BERT classification)
- **Bidirectional Attention**: Classification tasks use `is_causal=False`
- **Sign Correction**: QR decomposition requires sign correction for deterministic output: `q = q * signs.unsqueeze(0)`
