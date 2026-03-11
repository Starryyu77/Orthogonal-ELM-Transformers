#!/bin/bash
# Phase 6 Experiment Monitor
# Usage: ./monitor_phase6.sh [live]

cd /projects/LlamaFactory/OELM-Pretrain

LOGS_DIR="logs"
OUTPUTS_DIR="outputs/phase6_multidata"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Phase 6 Multi-Dataset Experiment Monitor"
echo "=========================================="
echo ""

# Check running jobs
echo "📊 Running Jobs:"
echo "----------------"
squeue -u tianyu016 --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "No jobs running or squeue not available"
echo ""

# Define datasets and methods
DATASETS=("ag_news" "sst2" "xnli" "mnli")
METHODS=("baseline" "oelm_qk" "oelm_qk_ffn")

echo "📁 Experiment Status:"
echo "---------------------"
printf "%-15s %-15s %-10s %-15s\n" "Dataset" "Method" "Status" "Best Accuracy"
echo "----------------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        RESULT_FILE="${OUTPUTS_DIR}/${dataset}/${method}/results.json"
        
        if [ -f "$RESULT_FILE" ]; then
            # Extract best accuracy from results
            BEST_ACC=$(python3 -c "import json; data=json.load(open('$RESULT_FILE')); accs=[r['accuracy'] for r in data]; print(f'{max(accs)*100:.2f}%')" 2>/dev/null || echo "N/A")
            STATUS="${GREEN}✅ Complete${NC}"
        else
            BEST_ACC="N/A"
            # Check if log file exists
            LOG_PATTERN="${LOGS_DIR}/${dataset%_*}_${method#oelm_}*.out"
            if ls $LOG_PATTERN 1> /dev/null 2>&1; then
                STATUS="${YELLOW}🔄 Running${NC}"
            else
                STATUS="${RED}⏳ Pending${NC}"
            fi
        fi
        
        printf "%-15s %-15s %-20s %-15s\n" "$dataset" "$method" "$STATUS" "$BEST_ACC"
    done
done

echo ""
echo "=========================================="

# Live mode - refresh every 10 seconds
if [ "$1" == "live" ]; then
    while true; do
        sleep 10
        clear
        $0
    done
fi
