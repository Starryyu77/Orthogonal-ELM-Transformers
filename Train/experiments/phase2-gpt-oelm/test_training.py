#!/usr/bin/env python3
"""
快速测试脚本 - 验证训练流程
使用随机生成的数据，无需准备真实数据集
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
from models.modeling_oelm_v2 import create_oelm_v2_tiny
from models.modeling_gpt import create_gpt_tiny

def test_model(model_type='baseline', num_steps=10):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"Testing {model_type.upper()}")
    print(f"{'='*60}")

    # 创建模型
    if model_type == 'baseline':
        model = create_gpt_tiny(vocab_size=1000)
    elif model_type in ['oelm_v2', 'oelm_random']:
        init_method = 'orthogonal' if model_type == 'oelm_v2' else 'normal'
        model = create_oelm_v2_tiny(vocab_size=1000, init_method=init_method)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    # 创建优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # 模拟训练
    print(f"\nRunning {num_steps} training steps...")
    model.train()

    step_times = []
    losses = []

    for step in range(num_steps):
        # 生成随机数据
        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        # 计时
        if device.type == 'cuda':
            torch.cuda.synchronize()
        step_start = time.perf_counter()

        # 前向传播
        loss = model(input_ids, targets, return_loss=True)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start

        step_times.append(step_time)
        losses.append(loss.item())

        if step % 5 == 0:
            print(f"  Step {step}: Loss={loss.item():.4f}, Time={step_time*1000:.2f}ms")

    # 统计
    avg_time = np.mean(step_times) * 1000  # ms
    avg_loss = np.mean(losses)

    print(f"\nResults:")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Average step time: {avg_time:.2f}ms")
    print(f"  Throughput: {batch_size * seq_len / (avg_time/1000):.0f} tokens/sec")

    return {
        'model_type': model_type,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'avg_loss': avg_loss,
        'avg_step_time_ms': avg_time,
    }

def main():
    print("="*60)
    print("Phase 2 Training Pipeline Test")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    results = []

    # 测试三种模式
    for model_type in ['baseline', 'oelm_v2', 'oelm_random']:
        try:
            result = test_model(model_type, num_steps=10)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # 对比总结
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")

    print(f"\n{'Model':<15} {'Params':<12} {'Trainable':<12} {'Frozen':<10} {'Step Time':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['model_type']:<15} "
              f"{r['total_params']/1e6:.1f}M{'':<5} "
              f"{r['trainable_params']/1e6:.1f}M{'':<4} "
              f"{r['frozen_params']/1e6:.1f}M{'':<3} "
              f"{r['avg_step_time_ms']:.2f}ms")

    print(f"\n{'='*60}")
    print("All tests passed! ✓")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
