#!/bin/bash
# Submit all fine-tuning jobs after pretraining completes
# Usage: ./submit_all_finetune.sh

echo "========================================"
echo "Submitting All Fine-tuning Jobs"
echo "========================================"

cd /projects/LlamaFactory/OELM-Pretrain

# Check if pretrained models exist
if [ ! -f "outputs/pretrain/baseline/best/pytorch_model.pt" ]; then
    echo "Error: Baseline pretrained model not found!"
    echo "Please wait for pretraining to complete."
    exit 1
fi

if [ ! -f "outputs/pretrain/oelm_qk/best/pytorch_model.pt" ]; then
    echo "Warning: OELM-QK pretrained model not found"
fi

if [ ! -f "outputs/pretrain/oelm_qk_ffn/best/pytorch_model.pt" ]; then
    echo "Warning: OELM-QK-FFN pretrained model not found"
fi

echo ""
echo "Submitting fine-tuning jobs..."

# Submit jobs
if [ -f "outputs/pretrain/baseline/best/pytorch_model.pt" ]; then
    JOB1=$(sbatch scripts/run_finetune_baseline.sh | awk '{print $4}')
    echo "  Baseline fine-tune: Job ID $JOB1"
fi

if [ -f "outputs/pretrain/oelm_qk/best/pytorch_model.pt" ]; then
    JOB2=$(sbatch scripts/run_finetune_qk.sh | awk '{print $4}')
    echo "  OELM-QK fine-tune: Job ID $JOB2"
fi

if [ -f "outputs/pretrain/oelm_qk_ffn/best/pytorch_model.pt" ]; then
    JOB3=$(sbatch scripts/run_finetune_qk_ffn.sh | awk '{print $4}')
    echo "  OELM-QK-FFN fine-tune: Job ID $JOB3"
fi

echo ""
echo "========================================"
echo "All fine-tuning jobs submitted!"
echo "========================================"