#!/usr/bin/env python3
"""
Training Script v2 for Head-wise Orthogonal ELM Transformer

支持:
- Head-wise Orthogonal ELM (v2) - 分头正交初始化
- Baseline GPT - 标准训练
- 严格计时记录 (CUDA synchronized)
- 多数据集: TinyStories, OpenWebText, WikiText-103

Usage:
    # Baseline GPT
    python train_v2.py --model_type baseline --dataset tinystories

    # OELM v2 (Head-wise orthogonal)
    python train_v2.py --model_type oelm_v2 --dataset tinystories

    # Multi-GPU
    torchrun --nproc_per_node=2 train_v2.py --model_type oelm_v2 --dataset tinystories
"""

import os
import sys
import time
import math
import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from tqdm import tqdm

# Import models
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import GPT, create_gpt2_small
from models.modeling_oelm_v2 import HeadWiseOrthogonalELMTransformer, create_oelm_v2_small


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def get_device(local_rank: int = 0):
    """Get the device to use for training."""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{local_rank}')
    return torch.device('cpu')


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for language modeling."""

    def __init__(self, data_path: str, seq_len: int = 512):
        self.seq_len = seq_len

        # Load tokenized data
        if data_path.endswith('.bin'):
            self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        elif data_path.endswith('.npy'):
            self.data = np.load(data_path, mmap_mode='r')
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        # Calculate number of sequences
        self.num_sequences = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1

        chunk = self.data[start_idx:end_idx].astype(np.int64)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float = 0.0):
    """Learning rate schedule with warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(args):
    """Main training function with detailed timing."""

    # ==========================================
    # Setup and initialization
    # ==========================================
    rank, world_size, local_rank = setup_distributed()
    device = get_device(local_rank)
    is_master = (rank == 0)

    # Random seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    # Create output directory
    if is_master:
        os.makedirs(args.out_dir, exist_ok=True)

    # ==========================================
    # Timing setup (CRITICAL)
    # ==========================================
    WARMUP_STEPS = 100  # Exclude first 100 steps from statistics
    step_times = []  # Pure training time per step
    validation_times = []  # Validation time
    epoch_times = []  # Time per epoch

    training_start_time = time.time()
    epoch_start_time = training_start_time

    # ==========================================
    # Load dataset
    # ==========================================
    if is_master:
        print(f"Loading dataset from {args.data_path}...")

    train_dataset = SimpleDataset(args.data_path, seq_len=args.seq_len)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Validation loader
    val_loader = None
    val_data_path = args.data_path.replace('train', 'val')
    if os.path.exists(val_data_path) and is_master:
        val_dataset = SimpleDataset(val_data_path, seq_len=args.seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # ==========================================
    # Create model
    # ==========================================
    if is_master:
        print(f"\nCreating {args.model_type} model...")

    if args.model_type == 'oelm_v2':
        model = HeadWiseOrthogonalELMTransformer(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            dropout=args.dropout,
            freeze_qk=True,
            init_method='orthogonal'
        )
    elif args.model_type == 'oelm_random':
        # OELM-Random: Random init + frozen Q/K (for ablation study)
        model = HeadWiseOrthogonalELMTransformer(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            dropout=args.dropout,
            freeze_qk=True,
            init_method='normal'
        )
    elif args.model_type == 'baseline':
        model = GPT(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model = model.to(device)

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # ==========================================
    # Optimizer
    # ==========================================
    # Only optimize parameters that require grad
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    if is_master:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in params_to_optimize)
        frozen_params = total_params - trainable_params
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    # Gradient scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # ==========================================
    # Resume from checkpoint
    # ==========================================
    start_step = 0
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        if is_master:
            print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint.get('step', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # ==========================================
    # Training loop
    # ==========================================
    if is_master:
        print(f"\n{'='*60}")
        print(f"Starting training")
        print(f"  Model: {args.model_type}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Steps: {start_step} -> {args.max_steps}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Effective batch size: {args.batch_size * world_size}")
        print(f"{'='*60}\n")

    model.train()
    train_iter = iter(train_loader)

    # Progress bar (master only)
    pbar = tqdm(total=args.max_steps, initial=start_step, desc="Training", disable=not is_master)

    for step in range(start_step, args.max_steps):
        # Update learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get next batch
        try:
            batch = next(train_iter)
        except StopIteration:
            # End of epoch
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)

            if is_master:
                print(f"\nEpoch completed in {epoch_duration:.1f}s")

            # Restart iterator
            train_iter = iter(train_loader)
            batch = next(train_iter)
            epoch_start_time = time.time()

        # ==========================================
        # Training step with timing
        # ==========================================
        if device.type == 'cuda':
            torch.cuda.synchronize()

        step_start_time = time.perf_counter()

        # Forward pass
        x, y = batch
        x, y = x.to(device), y.to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = model(x, y, return_loss=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(x, y, return_loss=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        optimizer.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        step_end_time = time.perf_counter()
        step_duration = step_end_time - step_start_time

        # Collect step time after warmup
        if step > WARMUP_STEPS:
            step_times.append(step_duration)

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})

        # ==========================================
        # Validation
        # ==========================================
        if step > 0 and step % args.val_interval == 0 and val_loader is not None:
            if device.type == 'cuda':
                torch.cuda.synchronize()

            val_start_time = time.perf_counter()

            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for val_batch in val_loader:
                    x, y = val_batch
                    x, y = x.to(device), y.to(device)
                    loss = model(x, y, return_loss=True)
                    val_loss += loss.item()
                    val_batches += 1

            val_loss /= max(val_batches, 1)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            val_end_time = time.perf_counter()
            val_duration = val_end_time - val_start_time
            validation_times.append(val_duration)

            # Calculate perplexity
            perplexity = math.exp(val_loss)

            if is_master:
                print(f"\n[Step {step}] Validation - Loss: {val_loss:.4f}, PPL: {perplexity:.2f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_to_save = model.module if isinstance(model, DDP) else model
                    checkpoint = {
                        'model': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'best_val_loss': best_val_loss,
                        'args': vars(args)
                    }
                    torch.save(checkpoint, os.path.join(args.out_dir, 'best.pt'))
                    print(f"  ✓ New best model saved (PPL: {perplexity:.2f})")

            model.train()

        # ==========================================
        # Periodic checkpointing
        # ==========================================
        if step > 0 and step % args.save_interval == 0 and is_master:
            model_to_save = model.module if isinstance(model, DDP) else model
            checkpoint = {
                'model': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.out_dir, 'latest.pt'))

    pbar.close()

    # ==========================================
    # Training completed - Save timing data
    # ==========================================
    if is_master:
        total_time = time.time() - training_start_time

        # Calculate timing statistics
        if step_times:
            step_times_array = np.array(step_times)
            median_time = np.median(step_times_array)
            filtered_times = step_times_array[step_times_array <= 3 * median_time]

            # Determine init_method based on model_type
            if args.model_type == 'oelm_v2':
                init_method = 'orthogonal'
            elif args.model_type == 'oelm_random':
                init_method = 'normal'
            else:
                init_method = 'n/a'

            timing_stats = {
                'model_type': args.model_type,
                'init_method': init_method,
                'dataset': args.dataset,
                'total_wall_time': total_time,
                'total_formatted': f"{int(total_time//3600)}h {int((total_time%3600)//60)}m {int(total_time%60)}s",
                'num_steps_measured': len(step_times),
                'warmup_steps': WARMUP_STEPS,
                'mean_step_time': float(np.mean(filtered_times)),
                'std_step_time': float(np.std(filtered_times)),
                'median_step_time': float(np.median(filtered_times)),
                'min_step_time': float(np.min(filtered_times)),
                'max_step_time': float(np.max(filtered_times)),
                'p95_step_time': float(np.percentile(filtered_times, 95)),
                'p99_step_time': float(np.percentile(filtered_times, 99)),
                'total_validation_time': float(np.sum(validation_times)),
                'mean_validation_time': float(np.mean(validation_times)) if validation_times else 0,
                'num_validations': len(validation_times),
                'epoch_times': epoch_times,
                'final_perplexity': math.exp(best_val_loss) if best_val_loss != float('inf') else None,
                'best_val_loss': best_val_loss if best_val_loss != float('inf') else None,
            }
        else:
            timing_stats = {
                'model_type': args.model_type,
                'dataset': args.dataset,
                'total_wall_time': total_time,
                'note': 'No timing data collected (training too short)'
            }

        # Save timing stats
        timing_file = os.path.join(args.out_dir, 'timing_stats.json')
        with open(timing_file, 'w') as f:
            json.dump(timing_stats, f, indent=2)

        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")
        print(f"Total time: {timing_stats['total_formatted']}")
        print(f"Best val loss: {best_val_loss:.4f}")
        if best_val_loss != float('inf'):
            print(f"Best perplexity: {math.exp(best_val_loss):.2f}")
        print(f"Timing stats saved to: {timing_file}")
        print(f"{'='*60}\n")

    # Cleanup
    if world_size > 1:
        destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train GPT with Head-wise Orthogonal ELM")

    # Model args
    parser.add_argument('--model_type', type=str, default='oelm_v2',
                        choices=['baseline', 'oelm_v2', 'oelm_random'],
                        help='Model type: baseline (GPT), oelm_v2 (Head-wise Orthogonal), or oelm_random (Random init + frozen Q/K)')
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Data args
    parser.add_argument('--dataset', type=str, default='tinystories',
                        choices=['tinystories', 'openwebtext', 'wikitext103'])
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to tokenized data (auto-set if None)')

    # Training args
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Logging args
    parser.add_argument('--val_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    # System args
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')

    args = parser.parse_args()

    # Auto-set data path
    if args.data_path is None:
        data_dir = Path(__file__).parent.parent / 'data' / args.dataset
        args.data_path = str(data_dir / 'train.bin')

    # Auto-set output directory
    if args.out_dir is None:
        out_base = Path(__file__).parent.parent / 'outputs'
        args.out_dir = str(out_base / f'{args.model_type}_{args.dataset}')

    # Check data exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data not found at {args.data_path}")
        print("Please run data preparation first:")
        print(f"  python data/prepare_data.py --dataset {args.dataset}")
        return

    train(args)


if __name__ == '__main__':
    main()
