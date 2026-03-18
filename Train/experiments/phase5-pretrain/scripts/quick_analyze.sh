#!/bin/bash
# Quick analysis of experiment results

cd /projects/LlamaFactory/OELM-Pretrain

echo "========================================"
echo "OELM Pretrain Quick Analysis"
echo "========================================"

echo ""
echo "Pretrained Models:"
echo "------------------"
for method in baseline oelm_qk oelm_qk_ffn; do
    path="outputs/pretrain/$method/best/pytorch_model.pt"
    if [ -f "$path" ]; then
        echo "✅ $method: $(ls -lh $path | awk '{print $5}')"
    else
        echo "⏳ $method: Not ready"
    fi
done

echo ""
echo "Fine-tuned Models:"
echo "------------------"
for method in baseline oelm_qk oelm_qk_ffn; do
    path="outputs/finetune/${method}_imdb/results.json"
    if [ -f "$path" ]; then
        best_acc=$(python3 -c "import json; data=json.load(open('$path')); best=max(data, key=lambda x: x['accuracy']); print(f\"{best['accuracy']*100:.2f}%\")" 2>/dev/null)
        echo "✅ $method: $best_acc"
    else
        echo "⏳ $method: Not ready"
    fi
done

echo ""
echo "========================================"
echo "To run detailed analysis:"
echo "  python3 scripts/analyze_results.py"
echo "========================================"