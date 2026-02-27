#!/bin/bash
# Auto-restart OELM-Freeze when Baseline completes

PROJECT_DIR="/usr1/home/s125mdg43_10/Orthogonal_ELM_Transformers/Train/bert-reservoir-project"
OUTPUT_BASE="${PROJECT_DIR}/outputs/xnli"
BASELINE_LOG="${OUTPUT_BASE}/baseline/training.log"
OELM_LOG="${OUTPUT_BASE}/oelm_freeze/training_restart.log"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Auto-restart Monitor for OELM-Freeze"
echo "========================================"
echo ""
echo "Monitoring: ${BASELINE_LOG}"
echo "Will start: OELM-Freeze on GPU 3"
echo "Config: batch_size=8, grad_accum=2"
echo ""

# Check interval (seconds)
CHECK_INTERVAL=300  # Check every 5 minutes

while true; do
    # Check if Baseline log exists
    if [ ! -f "$BASELINE_LOG" ]; then
        echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} Waiting for Baseline log..."
        sleep $CHECK_INTERVAL
        continue
    fi

    # Check if training completed
    if grep -q "Training completed" "$BASELINE_LOG" 2>/dev/null; then
        echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ✓ Baseline training completed!"
        echo ""

        # Show final results
        echo "Baseline Final Results:"
        grep "Best accuracy" "$BASELINE_LOG" | tail -1
        grep "TRAINING TIME" -A 10 "$BASELINE_LOG" | tail -11
        echo ""

        # Check if OELM is already running
        if tmux has-session -t xnli_oelm 2>/dev/null; then
            echo "OELM-Freeze session already exists. Stopping it first..."
            tmux kill-session -t xnli_oelm
        fi

        # Start OELM-Freeze
        echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} Starting OELM-Freeze..."
        echo ""

        tmux new-session -d -s xnli_oelm

        tmux send-keys -t xnli_oelm "cd ${PROJECT_DIR}" C-m
        tmux send-keys -t xnli_oelm "export CUDA_VISIBLE_DEVICES=3" C-m
        tmux send-keys -t xnli_oelm "export PYTHONPATH=${PROJECT_DIR}:$PYTHONPATH" C-m

        tmux send-keys -t xnli_oelm "python3 models/train_bert.py \\" C-m
        tmux send-keys -t xnli_oelm "  --dataset xnli \\" C-m
        tmux send-keys -t xnli_oelm "  --freeze_mode true \\" C-m
        tmux send-keys -t xnli_oelm "  --init_method orthogonal \\" C-m
        tmux send-keys -t xnli_oelm "  --lr 1e-4 \\" C-m
        tmux send-keys -t xnli_oelm "  --batch_size 8 \\" C-m
        tmux send-keys -t xnli_oelm "  --gradient_accumulation_steps 2 \\" C-m
        tmux send-keys -t xnli_oelm "  --epochs 3 \\" C-m
        tmux send-keys -t xnli_oelm "  --max_length 128 \\" C-m
        tmux send-keys -t xnli_oelm "  --validate_steps 500 \\" C-m
        tmux send-keys -t xnli_oelm "  --output_dir ${OUTPUT_BASE}/oelm_freeze \\" C-m
        tmux send-keys -t xnli_oelm "  --seed 42 \\" C-m
        tmux send-keys -t xnli_oelm "  2>&1 | tee ${OELM_LOG}" C-m

        echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ✓ OELM-Freeze started!"
        echo ""
        echo "Session: xnli_oelm (GPU 3)"
        echo "Log: ${OELM_LOG}"
        echo "Effective batch size: 8 * 2 = 16"
        echo ""
        echo "Monitor with: tmux attach -t xnli_oelm"
        echo ""

        # Exit the monitor script
        exit 0
    fi

    # Show current progress
    CURRENT_STEP=$(grep -o "Step [0-9]\+: Running validation" "$BASELINE_LOG" 2>/dev/null | tail -1 | grep -o "[0-9]\+")
    BEST_ACC=$(grep "New best model" "$BASELINE_LOG" 2>/dev/null | tail -1 | grep -o "Accuracy: [0-9.]\+" | cut -d' ' -f2)

    if [ -n "$CURRENT_STEP" ] && [ -n "$BEST_ACC" ]; then
        PROGRESS=$(echo "scale=1; $CURRENT_STEP / 73632 * 100" | bc -l 2>/dev/null || echo "N/A")
        echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} Baseline: Step $CURRENT_STEP ($PROGRESS%), Best Acc: $BEST_ACC - Still running..."
    else
        echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} Monitoring..."
    fi

    sleep $CHECK_INTERVAL
done
