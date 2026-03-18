#!/bin/bash
# Master script to submit all pretrain experiments
# Usage: ./submit_all.sh

set -e

echo "========================================"
echo "OELM Pretrain Experiment Submission"
echo "========================================"

# Check if we're on the cluster
if ! command -v sbatch &> /dev/null; then
    echo "Error: sbatch not found. This script must be run on the NTU cluster."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Submit jobs
echo ""
echo "Submitting pretrain jobs..."

JOB1=$(sbatch scripts/run_pretrain_baseline.sh | awk '{print $4}')
echo "  Baseline: Job ID $JOB1"

JOB2=$(sbatch scripts/run_pretrain_oelm_qk.sh | awk '{print $4}')
echo "  OELM-QK: Job ID $JOB2"

JOB3=$(sbatch scripts/run_pretrain_oelm_qk_ffn.sh | awk '{print $4}')
echo "  OELM-QK-FFN: Job ID $JOB3"

echo ""
echo "========================================"
echo "All jobs submitted!"
echo "========================================"
echo ""
echo "Monitor jobs with:"
echo "  watch squeue -u tianyu016"
echo ""
echo "View logs with:"
echo "  tail -f logs/pretrain_*.out"
echo ""
echo "Once pretrain completes, run fine-tuning:"
echo "  sbatch scripts/run_finetune.sh <checkpoint_path> <dataset> <method>"
echo ""
echo "Example:"
echo "  sbatch scripts/run_finetune.sh outputs/pretrain/oelm_qk/best/pytorch_model.pt imdb oelm_qk"