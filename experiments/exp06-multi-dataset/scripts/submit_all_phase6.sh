#!/bin/bash
# Phase 6 Multi-Dataset Validation - Master Submit Script
# Usage: ./submit_all_phase6.sh

cd /projects/LlamaFactory/OELM-Pretrain

echo "=========================================="
echo "Phase 6: Multi-Dataset Validation"
echo "Submitting all experiments..."
echo "Started at: $(date)"
echo "=========================================="

# Make sure logs directory exists
mkdir -p logs

# Function to submit job with dependency
submit_job() {
    local script=$1
    local name=$2
    echo "Submitting: $name"
    sbatch "$script"
    sleep 1  # Avoid overwhelming the scheduler
}

echo ""
echo ">>> AG News (4-class news classification)"
submit_job "scripts/run_agnews_baseline.sh" "AG News Baseline"
submit_job "scripts/run_agnews_qk.sh" "AG News OELM-QK"
submit_job "scripts/run_agnews_qk_ffn.sh" "AG News OELM-QK-FFN"

echo ""
echo ">>> SST-2 (2-class sentiment, short text)"
submit_job "scripts/run_sst2_baseline.sh" "SST-2 Baseline"
submit_job "scripts/run_sst2_qk.sh" "SST-2 OELM-QK"
submit_job "scripts/run_sst2_qk_ffn.sh" "SST-2 OELM-QK-FFN"

echo ""
echo ">>> XNLI (3-class NLI)"
submit_job "scripts/run_xnli_baseline.sh" "XNLI Baseline"
submit_job "scripts/run_xnli_qk.sh" "XNLI OELM-QK"
submit_job "scripts/run_xnli_qk_ffn.sh" "XNLI OELM-QK-FFN"

echo ""
echo ">>> MNLI (3-class large-scale NLI)"
submit_job "scripts/run_mnli_baseline.sh" "MNLI Baseline"
submit_job "scripts/run_mnli_qk.sh" "MNLI OELM-QK"
submit_job "scripts/run_mnli_qk_ffn.sh" "MNLI OELM-QK-FFN"

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Check status with: squeue -u tianyu016"
echo "=========================================="

# Show current queue
sleep 2
echo ""
echo "Current job queue:"
squeue -u tianyu016
