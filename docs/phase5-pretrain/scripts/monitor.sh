#!/bin/bash
# Monitor OELM pretrain experiments
# Usage: ./monitor.sh

echo "========================================"
echo "OELM Pretrain Experiment Monitor"
echo "========================================"
echo ""

# Check job status
echo "Job Status:"
squeue -u tianyu016 -o "%.10i %.15P %.20j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "  No jobs running"

echo ""
echo "========================================"
echo "Recent Logs:"
echo "========================================"

# Check log files
LOG_DIR="/projects/LlamaFactory/OELM-Pretrain/logs"
cd $LOG_DIR 2>/dev/null || exit 1

for log in $(ls -t *.out 2>/dev/null | head -5); do
    echo ""
    echo "--- $log ---"
    tail -20 "$log" 2>/dev/null
done

echo ""
echo "========================================"
echo "Checkpoint Status:"
echo "========================================"

# Check checkpoints
OUTPUT_DIR="/projects/LlamaFactory/OELM-Pretrain/outputs/pretrain"
if [ -d "$OUTPUT_DIR" ]; then
    for method in baseline oelm_qk oelm_qk_ffn; do
        checkpoint_dir="$OUTPUT_DIR/$method"
        if [ -d "$checkpoint_dir" ]; then
            latest=$(ls -t "$checkpoint_dir"/checkpoint-* 2>/dev/null | head -1)
            if [ -n "$latest" ]; then
                echo "$method: $(basename $latest)"
            else
                echo "$method: No checkpoints yet"
            fi
        else
            echo "$method: Directory not created"
        fi
    done
else
    echo "No outputs directory yet"
fi

echo ""
echo "========================================"
echo "GPU Usage:"
echo "========================================"
sinfo -o "%10N %10G %10C %10e" | grep -E "(NODELIST|pro6000|6000ada)" 2>/dev/null || echo "  GPU info unavailable"

echo ""
echo "Last updated: $(date)"
echo "========================================"