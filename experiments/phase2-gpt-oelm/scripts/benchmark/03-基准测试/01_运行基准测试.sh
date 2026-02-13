#!/bin/bash
# Run comprehensive benchmark suite

echo "=========================================="
echo "Running Benchmark Suite"
echo "=========================================="

MODEL_SIZE=${MODEL_SIZE:-"small"}

echo "Model Size: $MODEL_SIZE"
echo ""

# Benchmark OELM
echo "Benchmarking Orthogonal ELM Transformer..."
python benchmark.py \
    --model_type oelm \
    --model_size $MODEL_SIZE \
    --batch_size 16 \
    --seq_len 512 \
    --num_steps 100 \
    --memory_scaling

# Clear GPU cache
if command -v nvidia-smi &> /dev/null; then
    echo "Clearing GPU cache..."
fi

# Benchmark GPT
echo ""
echo "Benchmarking Standard GPT..."
python benchmark.py \
    --model_type gpt \
    --model_size $MODEL_SIZE \
    --batch_size 16 \
    --seq_len 512 \
    --num_steps 100 \
    --memory_scaling

# Run comparison
echo ""
echo "Running comparison..."
python benchmark.py --compare --model_size $MODEL_SIZE

echo ""
echo "Benchmark complete!"
