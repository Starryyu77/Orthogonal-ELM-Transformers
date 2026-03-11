# Phase 6: Multi-Dataset Validation Report

**Experiment**: Phase 6 Multi-Dataset Validation  
**Date**: 2026-03-11 ~ 2026-03-25  
**Location**: NTU EEE GPU Cluster  
**Status**: 🟡 In Progress / ⏳ Pending Results  

---

## 🎯 Executive Summary

### Research Question
Does OELM-QK-FFN generalize across different classification datasets?

### Key Findings (To be filled after experiments complete)

| Metric | Value |
|:-------|:------|
| **Datasets Validated** | X / 4 |
| **Best Method** | TBD |
| **Average Improvement** | TBD |
| **Parameter Efficiency** | 43.1% (OELM-QK-FFN) |

---

## 📊 Results Summary

### Overall Results Table

| Dataset | Classes | Baseline | OELM-QK | OELM-QK-FFN | Best | vs Baseline |
|:--------|:-------:|:--------:|:-------:|:-----------:|:----:|:-----------:|
| IMDB (Phase 5) | 2 | 82.95% | 84.06% | **84.78%** | QK-FFN | +1.83% |
| AG News | 4 | ☐ | ☐ | ☐ | - | - |
| SST-2 | 2 | ☐ | ☐ | ☐ | - | - |
| XNLI | 3 | ☐ | ☐ | ☐ | - | - |
| MNLI | 3 | ☐ | ☐ | ☐ | - | - |
| **Average** | - | - | - | - | - | - |

### Statistical Analysis

- **Paired t-test**: TBD
- **Effect size (Cohen's d)**: TBD
- **95% Confidence Interval**: TBD

---

## 📈 Detailed Results by Dataset

### AG News (4-Class News Classification)

**Dataset Info**:
- Classes: 4 (World, Sports, Business, Sci/Tech)
- Train: 120,000 samples
- Test: 7,600 samples

**Results**:

| Method | Best Acc | Best Epoch | Final Loss | F1 Score |
|:-------|:--------:|:----------:|:----------:|:--------:|
| Baseline | ☐ | - | - | - |
| OELM-QK | ☐ | - | - | - |
| OELM-QK-FFN | ☐ | - | - | - |

**Training Curves**:
```
(To be added after experiments)
```

---

### SST-2 (2-Class Sentiment Analysis)

**Dataset Info**:
- Classes: 2 (Positive, Negative)
- Train: 67,349 samples
- Test: 872 samples
- Characteristic: Short text (movie reviews)

**Results**:

| Method | Best Acc | Best Epoch | Final Loss | F1 Score |
|:-------|:--------:|:----------:|:----------:|:--------:|
| Baseline | ☐ | - | - | - |
| OELM-QK | ☐ | - | - | - |
| OELM-QK-FFN | ☐ | - | - | - |

---

### XNLI (3-Class Natural Language Inference)

**Dataset Info**:
- Classes: 3 (Entailment, Neutral, Contradiction)
- Train: 392,702 samples
- Validation: 2,490 samples
- Characteristic: Complex reasoning task

**Results**:

| Method | Best Acc | Best Epoch | Final Loss | F1 Score |
|:-------|:--------:|:----------:|:----------:|:--------:|
| Baseline | ☐ | - | - | - |
| OELM-QK | ☐ | - | - | - |
| OELM-QK-FFN | ☐ | - | - | - |

**Note**: XNLI showed +11.60% improvement in Phase 4 (GPT from scratch).

---

### MNLI (3-Class Large-Scale NLI)

**Dataset Info**:
- Classes: 3 (Entailment, Neutral, Contradiction)
- Train: 392,702 samples
- Validation (matched): 9,815 samples
- Characteristic: Larger validation set than XNLI

**Results**:

| Method | Best Acc | Best Epoch | Final Loss | F1 Score |
|:-------|:--------:|:----------:|:----------:|:--------:|
| Baseline | ☐ | - | - | - |
| OELM-QK | ☐ | - | - | - |
| OELM-QK-FFN | ☐ | - | - | - |

---

## 🔬 Analysis

### Cross-Dataset Performance

```
(To be filled after results collected)
- Performance by task type (binary vs multi-class)
- Performance by dataset size
- Performance by text length
```

### Parameter Efficiency Analysis

| Method | Trainable Params | Relative | Avg Accuracy | Efficiency Score |
|:-------|:----------------:|:--------:|:------------:|:----------------:|
| Baseline | 124.4M | 100% | - | 1.0x |
| OELM-QK | 110.2M | 88.6% | - | - |
| OELM-QK-FFN | 53.6M | 43.1% | - | - |

**Efficiency Score** = Accuracy / Trainable Params ratio

---

## ✅ Success Criteria Evaluation

| Criterion | Target | Achieved | Status |
|:----------|:-------|:---------|:------:|
| Multi-dataset validation | ≥4 datasets | X/4 | 🟡 |
| Performance maintenance | ≥Baseline-1% | TBD | 🟡 |
| Parameter efficiency | ≤50% params | 43.1% | ✅ |
| Statistical significance | p<0.05 | TBD | 🟡 |

---

## 📋 Experimental Setup

### Model Configuration

| Parameter | Value |
|:----------|:------|
| Architecture | GPT-Small |
| d_model | 768 |
| num_layers | 12 |
| num_heads | 12 |
| vocab_size | 50,257 |
| Total Parameters | 124.4M |

### Training Configuration

| Parameter | Baseline | OELM-QK | OELM-QK-FFN |
|:----------|:--------:|:-------:|:-----------:|
| Learning Rate | 3e-4 | 1e-3 | 1e-3 |
| Batch Size | 16 | 16 | 16 |
| Epochs | 3 | 3 | 3 |
| Max Seq Length | 512 | 512 | 512 |
| Warmup Steps | 500 | 500 | 500 |
| Weight Decay | 0.01 | 0.01 | 0.01 |

### Hardware

- GPU: NVIDIA RTX A5000 (24GB)
- CPU: 16 cores per job
- Cluster: NTU EEE GPU Cluster

---

## 🔍 Key Observations

### What Worked Well

(To be filled after experiments)

### Challenges

(To be filled after experiments)

### Surprising Findings

(To be filled after experiments)

---

## 📚 Comparison with Previous Phases

### Phase 4 (GPT from Scratch)

| Dataset | Phase 4 Result | Phase 6 Result | Difference |
|:--------|:--------------:|:--------------:|:----------:|
| IMDB | 78.56% → 85.70% (+7.14%) | 82.95% → 84.78% (+1.83%) | Pretraining effect |
| XNLI | 46.39% → 57.99% (+11.60%) | TBD | - |
| MNLI | 45.18% → 56.78% (+11.60%) | TBD | - |

### Phase 5 (Pretraining Validation)

| Dataset | Phase 5 Result | Phase 6 Result | Notes |
|:--------|:--------------:|:--------------:|:------|
| IMDB | 84.78% (QK-FFN) | - | Baseline for comparison |

---

## 🎯 Conclusions

(To be written after experiments complete)

### Main Conclusions

1. **Generalization**: TBD
2. **Parameter Efficiency**: TBD
3. **Task Dependencies**: TBD

### Implications

- **For Practitioners**: TBD
- **For Researchers**: TBD
- **For Industry**: TBD

---

## 🚀 Next Steps

### Immediate (Week 1-2)

- [ ] Complete all 12 experiments
- [ ] Collect and verify results
- [ ] Update this report with actual results

### Short-term (Month 1)

- [ ] Conduct ablation study (Phase 6.5)
- [ ] Statistical significance testing
- [ ] Paper draft preparation

### Long-term (Month 2-3)

- [ ] Scale to larger models (Phase 7)
- [ ] Additional tasks (NER, QA)
- [ ] Submit to conference

---

## 📁 Appendix

### A. Raw Results Files

```
outputs/phase6_multidata/
├── ag_news/
│   ├── baseline/results.json
│   ├── oelm_qk/results.json
│   └── oelm_qk_ffn/results.json
├── sst2/
│   ├── baseline/results.json
│   ├── oelm_qk/results.json
│   └── oelm_qk_ffn/results.json
├── xnli/
│   ├── baseline/results.json
│   ├── oelm_qk/results.json
│   └── oelm_qk_ffn/results.json
└── mnli/
    ├── baseline/results.json
    ├── oelm_qk/results.json
    └── oelm_qk_ffn/results.json
```

### B. Scripts Used

- `finetune_multidata.py`: Main training script
- `run_*_*.sh`: SLURM submission scripts
- `collect_results.py`: Results aggregation
- `monitor_phase6.sh`: Progress monitoring

### C. Reproducibility

To reproduce these experiments:

```bash
# Clone repository
git clone https://github.com/Starryyu77/Orthogonal-ELM-Transformers.git
cd Orthogonal-ELM-Transformers

# Run experiments
cd Train/experiments/phase6-multidata/scripts
./submit_all_phase6.sh

# Collect results
python collect_results.py
```

---

**Report Generated**: 2026-03-11  
**Last Updated**: 2026-03-11  
**Status**: 🟡 Awaiting experimental results  
**Next Update**: After experiments complete
