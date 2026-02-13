#!/bin/bash
# =============================================================================
# BERT Reservoir Test - Fair Comparison Experiment Controller
# =============================================================================
# This script runs multiple iterations of both Baseline and OELM-Freeze
# experiments under identical conditions for fair time comparison.
#
# Key Features (per 导师's Architecture Review):
#   1. AB-AB Interleaved Pattern: Baseline → OELM → Baseline → OELM ...
#      This controls for cluster state drift (temperature, load changes over time)
#   2. CUDA Synchronization: Uses torch.cuda.synchronize() for accurate GPU timing
#   3. Warmup Steps: First 100 steps excluded from statistics (CUDA context init)
#   4. High-Precision Timer: Uses time.perf_counter() for microsecond precision
#
# Usage:
#   ./run_fair_comparison.sh [num_runs]
#
# Arguments:
#   num_runs - Number of iterations to run (default: 3)
#
# Example:
#   ./run_fair_comparison.sh        # Run 3 iterations (6 total experiments)
#   ./run_fair_comparison.sh 5      # Run 5 iterations (10 total experiments)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NUM_RUNS=${1:-3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create output directories
mkdir -p outputs
mkdir -p logs
mkdir -p timing_results

# Timestamp for this experiment batch
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="logs/comparison_summary_${BATCH_TIMESTAMP}.txt"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}BERT Reservoir Test - Fair Comparison Experiment${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "Number of runs: ${NUM_RUNS}"
echo -e "Batch timestamp: ${BATCH_TIMESTAMP}"
echo -e "Summary file: ${SUMMARY_FILE}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Function to log to both console and file
log() {
    echo -e "$1"
    echo -e "$1" | sed 's/\x1b\[[0-9;]*m//g' >> "$SUMMARY_FILE"
}

# Function to check GPU availability
check_gpu() {
    log "${BLUE}Checking GPU status...${NC}"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader
    log ""
}

# Function to clear caches
clear_caches() {
    log "${YELLOW}Clearing caches...${NC}"

    # Clear Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    # Clear PyTorch CUDA cache
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    # Clear HuggingFace cache (optional, comment out if you want to keep it)
    # rm -rf ~/.cache/huggingface/datasets/ 2>/dev/null || true

    log "${GREEN}✓ Caches cleared${NC}"
    log ""
}

# Function to record system state
record_system_state() {
    local output_file=$1
    echo "System State - $(date)" > "$output_file"
    echo "========================================" >> "$output_file"
    echo "" >> "$output_file"
    echo "GPU Status:" >> "$output_file"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv >> "$output_file"
    echo "" >> "$output_file"
    echo "System Load:" >> "$output_file"
    uptime >> "$output_file"
    echo "" >> "$output_file"
    echo "Memory:" >> "$output_file"
    free -h >> "$output_file"
}

# Function to run a single experiment
run_experiment() {
    local mode=$1  # "baseline" or "oelm"
    local run_num=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)

    local log_file="logs/${mode}_run${run_num}_${timestamp}.log"
    local output_dir="outputs/${mode}_run${run_num}"

    if [ "$mode" == "baseline" ]; then
        freeze_mode="false"
        lr="2e-5"
        exp_name="Baseline"
    else
        freeze_mode="true"
        lr="1e-4"
        exp_name="OELM-Freeze"
    fi

    log "${GREEN}========================================${NC}"
    log "${GREEN}Run ${run_num}/${NUM_RUNS}: ${exp_name}${NC}"
    log "${GREEN}Start time: $(date)${NC}"
    log "${GREEN}Log file: ${log_file}${NC}"
    log "${GREEN}========================================${NC}"

    # Create output directory
    mkdir -p "$output_dir"

    # Record system state before run
    record_system_state "${output_dir}/system_state_before.txt"

    # Run the experiment
    python models/train_bert.py \
        --freeze_mode $freeze_mode \
        --lr $lr \
        --batch_size 32 \
        --epochs 3 \
        --output_dir "$output_dir" \
        --validate_steps 500 \
        2>&1 | tee "$log_file"

    # Record system state after run
    record_system_state "${output_dir}/system_state_after.txt"

    # Extract timing stats
    if [ -f "${output_dir}/timing_stats.json" ]; then
        cp "${output_dir}/timing_stats.json" "timing_results/${mode}_run${run_num}_${timestamp}.json"
        log "${GREEN}✓ Timing stats saved${NC}"
    fi

    log "${GREEN}✓ Run ${run_num} completed: ${exp_name}${NC}"
    log "${GREEN}End time: $(date)${NC}"
    log ""

    # Clear caches between runs
    clear_caches

    # Wait a bit for GPU to cool down
    log "${YELLOW}Waiting 10 seconds for GPU to stabilize...${NC}"
    sleep 10
}

# Function to analyze results
analyze_results() {
    log "${BLUE}============================================================${NC}"
    log "${BLUE}Analyzing Results${NC}"
    log "${BLUE}============================================================${NC}"

    python3 << 'PYTHON_SCRIPT'
import json
import glob
import numpy as np
from pathlib import Path

# Collect all timing results
baseline_files = sorted(glob.glob("timing_results/baseline_run*.json"))
oelm_files = sorted(glob.glob("timing_results/oelm_run*.json"))

def analyze_runs(files, name):
    if not files:
        print(f"No data found for {name}")
        return None

    mean_times = []
    pure_training_times = []
    total_wall_times = []

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            mean_times.append(data['mean_time_per_step'])
            pure_training_times.append(data['pure_training_time'])
            total_wall_times.append(data['total_wall_time'])

    return {
        'name': name,
        'n_runs': len(files),
        'mean_time_per_step': {
            'mean': np.mean(mean_times),
            'std': np.std(mean_times),
            'min': np.min(mean_times),
            'max': np.max(mean_times),
            'values': mean_times
        },
        'pure_training_time': {
            'mean': np.mean(pure_training_times),
            'std': np.std(pure_training_times)
        },
        'total_wall_time': {
            'mean': np.mean(total_wall_times),
            'std': np.std(total_wall_times)
        }
    }

baseline_stats = analyze_runs(baseline_files, "Baseline")
oelm_stats = analyze_runs(oelm_files, "OELM-Freeze")

print("\n" + "="*60)
print("FAIR COMPARISON RESULTS")
print("="*60)

if baseline_stats and oelm_stats:
    print("\nMean Time per Step (after warmup, excluding validation):")
    print(f"  Baseline:     {baseline_stats['mean_time_per_step']['mean']:.4f}s ± {baseline_stats['mean_time_per_step']['std']:.4f}s")
    print(f"  OELM-Freeze:  {oelm_stats['mean_time_per_step']['mean']:.4f}s ± {oelm_stats['mean_time_per_step']['std']:.4f}s")

    diff = oelm_stats['mean_time_per_step']['mean'] - baseline_stats['mean_time_per_step']['mean']
    pct_diff = (diff / baseline_stats['mean_time_per_step']['mean']) * 100
    print(f"\n  Difference: {diff:.4f}s ({pct_diff:+.1f}%)")

    print("\nPure Training Time (excluding validation):")
    print(f"  Baseline:     {baseline_stats['pure_training_time']['mean']:.1f}s ± {baseline_stats['pure_training_time']['std']:.1f}s")
    print(f"  OELM-Freeze:  {oelm_stats['pure_training_time']['mean']:.1f}s ± {oelm_stats['pure_training_time']['std']:.1f}s")

    print("\nTotal Wall Time:")
    print(f"  Baseline:     {baseline_stats['total_wall_time']['mean']:.1f}s ± {baseline_stats['total_wall_time']['std']:.1f}s")
    print(f"  OELM-Freeze:  {oelm_stats['total_wall_time']['mean']:.1f}s ± {oelm_stats['total_wall_time']['std']:.1f}s")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    if abs(pct_diff) < 5:
        print(f"✓ No significant difference in training speed ({abs(pct_diff):.1f}%)")
    elif pct_diff > 0:
        print(f"⚠ OELM-Freeze is {pct_diff:.1f}% slower than Baseline")
    else:
        print(f"⚠ OELM-Freeze is {abs(pct_diff):.1f}% faster than Baseline")

print("="*60)
PYTHON_SCRIPT

    log "${BLUE}============================================================${NC}"
}

# ============================================================================
# Main Execution
# ============================================================================

# Initialize summary file
echo "BERT Reservoir Test - Fair Comparison Experiment" > "$SUMMARY_FILE"
echo "Batch Timestamp: $BATCH_TIMESTAMP" >> "$SUMMARY_FILE"
echo "Number of Runs: $NUM_RUNS" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Check GPU
check_gpu

# Clear caches before starting
clear_caches

# Run experiments in AB-AB interleaved pattern (架构师建议 #3)
# This avoids systematic bias from changing cluster conditions
for i in $(seq 1 $NUM_RUNS); do
    log "${BLUE}============================================================${NC}"
    log "${BLUE}ITERATION ${i}/${NUM_RUNS} - AB-AB Interleaved Pattern${NC}"
    log "${BLUE}============================================================${NC}"
    log ""

    # Run A (Baseline) then B (OELM) in each iteration
    # This ensures both experiments experience similar cluster conditions
    run_experiment "baseline" $i
    run_experiment "oelm" $i

done

log "${BLUE}============================================================${NC}"
log "${BLUE}AB-AB Interleaved Pattern Complete${NC}"
log "${BLUE}Each iteration: Baseline → OELM-Freeze${NC}"
log "${BLUE}This controls for cluster state drift over time${NC}"
log "${BLUE}============================================================${NC}"
log ""

# Analyze results
analyze_results

# Final summary
log "${GREEN}============================================================${NC}"
log "${GREEN}All experiments completed!${NC}"
log "${GREEN}Summary file: ${SUMMARY_FILE}${NC}"
log "${GREEN}Timing results: timing_results/${NC}"
log "${GREEN}============================================================${NC}"

# Display final summary
cat "$SUMMARY_FILE"
