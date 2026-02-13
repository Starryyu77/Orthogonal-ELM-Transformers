#!/bin/bash
# BERT Reservoir Test Experiment Launcher
#
# This script launches the BERT OELM training on MLDA GPU cluster.
# It supports both Baseline (full fine-tuning) and OELM-Freeze modes.
#
# Usage:
#   ./run_experiment.sh baseline       # Run baseline with 2e-5 lr
#   ./run_experiment.sh oelm           # Run OELM with 1e-4 lr
#   ./run_experiment.sh quick-test     # Quick 100-step test
#   ./run_experiment.sh ddp-baseline   # Dual-GPU baseline
#   ./run_experiment.sh ddp-oelm       # Dual-GPU OELM

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
PROJECT_NAME="bert-reservoir-test"
OUTPUT_DIR="./outputs"
LOG_DIR="./logs"

# Create directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Function: print usage
print_usage() {
    echo -e "${BLUE}BERT Reservoir Test Experiment Launcher${NC}"
    echo ""
    echo "Usage: ./run_experiment.sh [MODE]"
    echo ""
    echo "Modes:"
    echo "  baseline       - Single GPU Baseline (freeze_mode=false, lr=2e-5)"
    echo "  oelm           - Single GPU OELM-Freeze (freeze_mode=true, lr=1e-4)"
    echo "  quick-test     - Quick 100-step test for both modes"
    echo "  ddp-baseline   - Dual GPU Baseline with DDP"
    echo "  ddp-oelm       - Dual GPU OELM-Freeze with DDP"
    echo "  compare        - Run both modes sequentially for comparison"
    echo "  lr-search      - Test different learning rates (100 steps each)"
    echo ""
    echo "Examples:"
    echo "  ./run_experiment.sh baseline"
    echo "  ./run_experiment.sh ddp-oelm"
    echo "  ./run_experiment.sh quick-test"
}

# Function: run training
run_training() {
    local freeze_mode=$1
    local lr=$2
    local batch_size=$3
    local max_steps=$4
    local ddp=$5
    local exp_name=$6

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="$LOG_DIR/${exp_name}_${timestamp}.log"

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Experiment: $exp_name${NC}"
    echo -e "${GREEN}Freeze Mode: $freeze_mode${NC}"
    echo -e "${GREEN}Learning Rate: $lr${NC}"
    echo -e "${GREEN}Batch Size: $batch_size${NC}"
    echo -e "${GREEN}Max Steps: $max_steps${NC}"
    echo -e "${GREEN}DDP: $ddp${NC}"
    echo -e "${GREEN}Log: $log_file${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Build command
    local cmd=""

    if [ "$ddp" = "true" ]; then
        # Dual GPU DDP
        cmd="torchrun --nproc_per_node=2 --master_port=29500 src/train_bert.py"
    else
        # Single GPU
        cmd="python src/train_bert.py"
    fi

    # Add arguments
    cmd="$cmd --freeze_mode $freeze_mode"
    cmd="$cmd --lr $lr"
    cmd="$cmd --batch_size $batch_size"
    cmd="$cmd --max_steps $max_steps"
    cmd="$cmd --output_dir $OUTPUT_DIR/$exp_name"
    cmd="$cmd --validate_steps 500"

    # Log the command
    echo "Command: $cmd" | tee "$log_file"
    echo "" | tee -a "$log_file"

    # Run with mlda gpu
    if command -v mlda &> /dev/null; then
        echo -e "${YELLOW}Using mlda gpu launcher...${NC}"
        mlda gpu $cmd 2>&1 | tee -a "$log_file"
    else
        echo -e "${YELLOW}mlda command not found, running directly...${NC}"
        $cmd 2>&1 | tee -a "$log_file"
    fi

    echo -e "${GREEN}✓ Experiment completed: $exp_name${NC}"
    echo -e "${GREEN}  Log saved to: $log_file${NC}"
    echo ""
}

# Main switch
case "${1:-}" in
    baseline)
        echo -e "${BLUE}Running Baseline (Full Fine-tuning)...${NC}"
        run_training "false" "2e-5" "32" "-1" "false" "baseline"
        ;;

    oelm)
        echo -e "${BLUE}Running OELM-Freeze...${NC}"
        run_training "true" "1e-4" "32" "-1" "false" "oelm_freeze"
        ;;

    quick-test)
        echo -e "${BLUE}Running Quick Test (100 steps)...${NC}"

        # Baseline quick test
        echo -e "${YELLOW}Test 1: Baseline (100 steps, lr=2e-5)${NC}"
        run_training "false" "2e-5" "32" "100" "false" "baseline_quick"

        echo -e "${YELLOW}Test 2: OELM (100 steps, lr=1e-4)${NC}"
        run_training "true" "1e-4" "32" "100" "false" "oelm_quick"

        echo -e "${GREEN}Quick tests completed!${NC}"
        ;;

    ddp-baseline)
        echo -e "${BLUE}Running Dual-GPU Baseline with DDP...${NC}"
        run_training "false" "2e-5" "32" "-1" "true" "ddp_baseline"
        ;;

    ddp-oelm)
        echo -e "${BLUE}Running Dual-GPU OELM-Freeze with DDP...${NC}"
        run_training "true" "1e-4" "32" "-1" "true" "ddp_oelm_freeze"
        ;;

    compare)
        echo -e "${BLUE}Running Comparison Experiment (Baseline vs OELM)...${NC}"

        # Run baseline first
        echo -e "${YELLOW}Step 1/2: Baseline Training${NC}"
        run_training "false" "2e-5" "32" "-1" "false" "compare_baseline"

        # Then OELM
        echo -e "${YELLOW}Step 2/2: OELM-Freeze Training${NC}"
        run_training "true" "1e-4" "32" "-1" "false" "compare_oelm"

        echo -e "${GREEN}Comparison experiment completed!${NC}"
        echo -e "${GREEN}Check $OUTPUT_DIR/compare_*/ for results${NC}"
        ;;

    lr-search)
        echo -e "${BLUE}Running Learning Rate Search...${NC}"

        # Test different learning rates for OELM mode
        for lr in "2e-5" "5e-5" "1e-4" "2e-4"; do
            echo -e "${YELLOW}Testing lr=$lr (100 steps)...${NC}"
            run_training "true" "$lr" "32" "100" "false" "lr_search_${lr}"
        done

        echo -e "${GREEN}Learning rate search completed!${NC}"
        echo -e "${GREEN}Check $LOG_DIR/lr_search_*.log for loss curves${NC}"
        ;;

    help|--help|-h)
        print_usage
        ;;

    *)
        echo -e "${RED}Error: Unknown mode '${1:-}'${NC}"
        print_usage
        exit 1
        ;;
esac
