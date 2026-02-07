#!/usr/bin/env python3
"""
Benchmark Script for Orthogonal ELM Transformer

This script benchmarks the performance of Orthogonal ELM Transformer
compared to standard GPT models.

Metrics measured:
- Training throughput (tokens/sec)
- Inference throughput (tokens/sec)
- Peak memory usage (GB)
- Parameter count

Usage:
    # Benchmark OELM model
    python benchmark.py --model_type oelm --model_size small
    
    # Benchmark GPT model
    python benchmark.py --model_type gpt --model_size small
    
    # Compare both
    python benchmark.py --compare --model_size small
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import (
    OrthogonalELMTransformer, create_oelm_tiny, create_oelm_small, create_oelm_medium,
    GPT, create_gpt_tiny, create_gpt2_small, create_gpt2_medium
)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters.
    
    Returns:
        (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_memory_usage() -> float:
    """Measure current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def benchmark_training_throughput(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_steps: int = 100,
    warmup_steps: int = 10
) -> Dict[str, float]:
    """
    Benchmark training throughput.
    
    Args:
        model: Model to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        num_steps: Number of steps to measure
        warmup_steps: Number of warmup steps
        
    Returns:
        Dictionary with throughput metrics
    """
    device = get_device()
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup
    for _ in range(warmup_steps):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        optimizer.zero_grad()
        loss = model(input_ids, targets, return_loss=True)
        loss.backward()
        optimizer.step()
    
    # Reset memory stats
    reset_memory_stats()
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(num_steps):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        optimizer.zero_grad()
        loss = model(input_ids, targets, return_loss=True)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    tokens_processed = batch_size * seq_len * num_steps
    throughput = tokens_processed / elapsed_time
    time_per_step = elapsed_time / num_steps
    peak_memory = measure_memory_usage()
    
    return {
        'throughput_tokens_per_sec': throughput,
        'time_per_step_ms': time_per_step * 1000,
        'peak_memory_gb': peak_memory,
        'batch_size': batch_size,
        'seq_len': seq_len
    }


def benchmark_inference_throughput(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_steps: int = 100,
    warmup_steps: int = 10
) -> Dict[str, float]:
    """
    Benchmark inference throughput.
    
    Args:
        model: Model to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        num_steps: Number of steps to measure
        warmup_steps: Number of warmup steps
        
    Returns:
        Dictionary with throughput metrics
    """
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            _ = model(input_ids, return_loss=False)
    
    # Reset memory stats
    reset_memory_stats()
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_steps):
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            _ = model(input_ids, return_loss=False)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    tokens_processed = batch_size * seq_len * num_steps
    throughput = tokens_processed / elapsed_time
    time_per_step = elapsed_time / num_steps
    peak_memory = measure_memory_usage()
    
    return {
        'throughput_tokens_per_sec': throughput,
        'time_per_step_ms': time_per_step * 1000,
        'peak_memory_gb': peak_memory,
        'batch_size': batch_size,
        'seq_len': seq_len
    }


def benchmark_memory_scaling(
    model: nn.Module,
    seq_len: int,
    batch_sizes: List[int]
) -> List[Dict[str, float]]:
    """
    Benchmark memory usage across different batch sizes.
    
    Args:
        model: Model to benchmark
        seq_len: Sequence length
        batch_sizes: List of batch sizes to test
        
    Returns:
        List of memory usage metrics for each batch size
    """
    device = get_device()
    model = model.to(device)
    model.train()
    
    results = []
    
    for batch_size in batch_sizes:
        reset_memory_stats()
        
        try:
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            targets = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            
            loss = model(input_ids, targets, return_loss=True)
            loss.backward()
            
            peak_memory = measure_memory_usage()
            
            results.append({
                'batch_size': batch_size,
                'peak_memory_gb': peak_memory,
                'success': True
            })
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                results.append({
                    'batch_size': batch_size,
                    'peak_memory_gb': None,
                    'success': False,
                    'error': 'OOM'
                })
                torch.cuda.empty_cache()
            else:
                raise e
    
    return results


def create_model(model_type: str, model_size: str, vocab_size: int = 50257):
    """Create a model based on type and size."""
    if model_type == 'oelm':
        if model_size == 'tiny':
            return create_oelm_tiny(vocab_size)
        elif model_size == 'small':
            return create_oelm_small(vocab_size)
        elif model_size == 'medium':
            return create_oelm_medium(vocab_size)
        else:
            raise ValueError(f"Unknown model size: {model_size}")
    elif model_type == 'gpt':
        if model_size == 'tiny':
            return create_gpt_tiny(vocab_size)
        elif model_size == 'small':
            return create_gpt2_small(vocab_size)
        elif model_size == 'medium':
            return create_gpt2_medium(vocab_size)
        else:
            raise ValueError(f"Unknown model size: {model_size}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_benchmark(args) -> Dict:
    """Run full benchmark suite."""
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    print(f"\nCreating {args.model_type.upper()} model (size: {args.model_size})...")
    model = create_model(args.model_type, args.model_size, args.vocab_size)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    
    results = {
        'model_type': args.model_type,
        'model_size': args.model_size,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_ratio': trainable_params / total_params
    }
    
    # Benchmark training throughput
    if not args.skip_training:
        print(f"\n{'='*60}")
        print("Benchmarking Training Throughput")
        print(f"{'='*60}")
        print(f"Batch size: {args.batch_size}, Sequence length: {args.seq_len}")
        
        train_results = benchmark_training_throughput(
            model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps
        )
        
        print(f"\nTraining Results:")
        print(f"  Throughput: {train_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
        print(f"  Time per step: {train_results['time_per_step_ms']:.2f} ms")
        print(f"  Peak memory: {train_results['peak_memory_gb']:.2f} GB")
        
        results['training'] = train_results
    
    # Benchmark inference throughput
    if not args.skip_inference:
        print(f"\n{'='*60}")
        print("Benchmarking Inference Throughput")
        print(f"{'='*60}")
        print(f"Batch size: {args.batch_size}, Sequence length: {args.seq_len}")
        
        infer_results = benchmark_inference_throughput(
            model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps
        )
        
        print(f"\nInference Results:")
        print(f"  Throughput: {infer_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
        print(f"  Time per step: {infer_results['time_per_step_ms']:.2f} ms")
        print(f"  Peak memory: {infer_results['peak_memory_gb']:.2f} GB")
        
        results['inference'] = infer_results
    
    # Benchmark memory scaling
    if args.memory_scaling:
        print(f"\n{'='*60}")
        print("Benchmarking Memory Scaling")
        print(f"{'='*60}")
        
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        memory_results = benchmark_memory_scaling(model, args.seq_len, batch_sizes)
        
        print(f"\nMemory Scaling Results:")
        for r in memory_results:
            if r['success']:
                print(f"  Batch size {r['batch_size']:2d}: {r['peak_memory_gb']:.2f} GB")
            else:
                print(f"  Batch size {r['batch_size']:2d}: OOM")
        
        results['memory_scaling'] = memory_results
    
    return results


def compare_models(args):
    """Compare OELM and GPT models."""
    print(f"\n{'='*80}")
    print("COMPARISON: Orthogonal ELM Transformer vs Standard GPT")
    print(f"{'='*80}")
    
    # Benchmark OELM
    args.model_type = 'oelm'
    print("\n" + "="*80)
    print("ORTHOGONAL ELM TRANSFORMER")
    print("="*80)
    oelm_results = run_benchmark(args)
    
    # Clear cache between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Benchmark GPT
    args.model_type = 'gpt'
    print("\n" + "="*80)
    print("STANDARD GPT")
    print("="*80)
    gpt_results = run_benchmark(args)
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nParameter Efficiency:")
    print(f"  OELM trainable: {oelm_results['trainable_ratio']*100:.1f}%")
    print(f"  GPT trainable: {gpt_results['trainable_ratio']*100:.1f}%")
    print(f"  Reduction: {(1 - oelm_results['trainable_ratio']/gpt_results['trainable_ratio'])*100:.1f}%")
    
    if 'training' in oelm_results and 'training' in gpt_results:
        oelm_train = oelm_results['training']
        gpt_train = gpt_results['training']
        
        print(f"\nTraining Speed:")
        print(f"  OELM: {oelm_train['throughput_tokens_per_sec']:,.0f} tokens/sec")
        print(f"  GPT:  {gpt_train['throughput_tokens_per_sec']:,.0f} tokens/sec")
        speedup = oelm_train['throughput_tokens_per_sec'] / gpt_train['throughput_tokens_per_sec']
        print(f"  Speedup: {speedup:.2f}x")
        
        print(f"\nTraining Memory:")
        print(f"  OELM: {oelm_train['peak_memory_gb']:.2f} GB")
        print(f"  GPT:  {gpt_train['peak_memory_gb']:.2f} GB")
        mem_reduction = (1 - oelm_train['peak_memory_gb'] / gpt_train['peak_memory_gb']) * 100
        print(f"  Reduction: {mem_reduction:.1f}%")
    
    # Save comparison results
    comparison = {
        'oelm': oelm_results,
        'gpt': gpt_results
    }
    
    output_path = f"benchmark_comparison_{args.model_size}.json"
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Orthogonal ELM Transformer')
    
    parser.add_argument('--model_type', type=str, default='oelm', choices=['oelm', 'gpt'],
                        help='Model type to benchmark')
    parser.add_argument('--model_size', type=str, default='small', choices=['tiny', 'small', 'medium'],
                        help='Model size')
    parser.add_argument('--vocab_size', type=int, default=50257,
                        help='Vocabulary size')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for benchmarking')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Sequence length')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of steps to benchmark')
    parser.add_argument('--warmup_steps', type=int, default=10,
                        help='Number of warmup steps')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training benchmark')
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference benchmark')
    parser.add_argument('--memory_scaling', action='store_true',
                        help='Run memory scaling benchmark')
    parser.add_argument('--compare', action='store_true',
                        help='Compare OELM and GPT models')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args)
    else:
        results = run_benchmark(args)
        
        # Save results
        output_path = f"benchmark_{args.model_type}_{args.model_size}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
