#!/usr/bin/env python3
"""
Training Script for Orthogonal ELM Transformer

This script trains both Orthogonal ELM Transformer and standard GPT models
for comparison. Supports distributed training with DDP and logging with WandB.

Usage:
    # Single GPU
    python train.py --model_type oelm --dataset tinystories
    
    # Multi-GPU DDP
    torchrun --nproc_per_node=4 train.py --model_type oelm --dataset tinystories
    
    # Resume training
    python train.py --resume out/latest.pt
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
from models import (
    OrthogonalELMTransformer, create_oelm_small,
    GPT, create_gpt2_small
)

# Try to import wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Logging to console only.")


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
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    
    if step > max_steps:
        return min_lr
    
    # Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_step(model, batch, device, scaler=None):
    """Single training step."""
    x, y = batch
    x, y = x.to(device), y.to(device)
    
    if scaler is not None:
        # Mixed precision training
        with torch.cuda.amp.autocast():
            loss = model(x, y, return_loss=True)
        scaler.scale(loss).backward()
    else:
        loss = model(x, y, return_loss=True)
        loss.backward()
    
    return loss.item()


def validate(model, dataloader, device, max_batches: Optional[int] = None):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            loss = model(x, y, return_loss=True)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)


def train(args):
    """Main training function."""
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = get_device(local_rank)
    is_master = (rank == 0)
    
    # Set random seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    # Create output directory
    if is_master:
        os.makedirs(args.out_dir, exist_ok=True)
    
    # Load dataset
    if is_master:
        print(f"Loading dataset from {args.data_path}...")
    
    train_dataset = SimpleDataset(args.data_path, seq_len=args.seq_len)
    
    # Create dataloader
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create validation dataloader if validation data exists
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
    
    # Create model
    if is_master:
        print(f"Creating {args.model_type} model...")
    
    if args.model_type == 'oelm':
        model = OrthogonalELMTransformer(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            dropout=args.dropout,
            ortho_method=args.ortho_method,
            freeze_qk=args.freeze_qk
        )
    elif args.model_type == 'gpt':
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
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (manual)
    # We'll handle this in the training loop
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    # Initialize wandb
    if is_master and HAS_WANDB and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{args.model_type}_{args.dataset}",
            config=vars(args)
        )
    
    # Resume from checkpoint if specified
    start_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if is_master:
            print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint.get('step', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Training loop
    if is_master:
        print(f"\nStarting training...")
        print(f"  Steps: {start_step} -> {args.max_steps}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  World size: {world_size}")
        print(f"  Effective batch size: {args.batch_size * world_size}")
        print()
    
    model.train()
    train_iter = iter(train_loader)
    
    for step in range(start_step, args.max_steps):
        # Update learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get next batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            if sampler is not None:
                sampler.set_epoch(step)
        
        # Training step
        optimizer.zero_grad()
        loss = train_step(model, batch, device, scaler)
        
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Logging
        if is_master and step % args.log_interval == 0:
            perplexity = math.exp(min(loss, 10))  # Cap to avoid overflow
            
            log_msg = f"Step {step:6d} | Loss: {loss:.4f} | PPL: {perplexity:.2f} | LR: {lr:.2e}"
            print(log_msg)
            
            if HAS_WANDB and args.use_wandb:
                wandb.log({
                    'train/loss': loss,
                    'train/perplexity': perplexity,
                    'train/lr': lr,
                    'train/step': step
                })
        
        # Validation
        if is_master and val_loader is not None and step % args.val_interval == 0 and step > 0:
            val_loss = validate(model, val_loader, device, max_batches=args.val_batches)
            val_ppl = math.exp(min(val_loss, 10))
            
            print(f"  Validation | Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")
            
            if HAS_WANDB and args.use_wandb:
                wandb.log({
                    'val/loss': val_loss,
                    'val/perplexity': val_ppl,
                    'val/step': step
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'best_val_loss': best_val_loss,
                    'args': vars(args)
                }
                torch.save(checkpoint, os.path.join(args.out_dir, 'best.pt'))
                print(f"  Saved best model (val_loss: {val_loss:.4f})")
            
            model.train()
        
        # Save checkpoint
        if is_master and step % args.save_interval == 0 and step > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.out_dir, 'latest.pt'))
        
        # Calculate MFU (Model FLOPs Utilization)
        if is_master and step % args.log_interval == 0:
            # Approximate FLOPs per token
            # Attention: 4 * d_model^2 (Q, K, V, O projections)
            # FFN: 8 * d_model^2 (2 linear layers with 4x expansion)
            # Total per layer: 12 * d_model^2
            # Total per forward pass: num_layers * 12 * d_model^2 * seq_len
            flops_per_token = args.num_layers * 12 * args.d_model * args.d_model
            flops_per_forward = flops_per_token * args.batch_size * args.seq_len
            
            # Estimate tokens per second
            # This is a rough estimate
            if step > start_step:
                # We'd need actual timing for accurate MFU
                pass
    
    # Save final checkpoint
    if is_master:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': args.max_steps,
            'best_val_loss': best_val_loss,
            'args': vars(args)
        }
        torch.save(checkpoint, os.path.join(args.out_dir, 'final.pt'))
        print(f"\nTraining complete! Final checkpoint saved.")
    
    # Cleanup
    if world_size > 1:
        destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Train Orthogonal ELM Transformer')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='oelm', choices=['oelm', 'gpt'],
                        help='Model type: oelm or gpt')
    parser.add_argument('--vocab_size', type=int, default=50257,
                        help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--ortho_method', type=str, default='qr', choices=['qr', 'svd', 'householder'],
                        help='Orthogonal initialization method (for OELM)')
    parser.add_argument('--freeze_qk', type=lambda x: x.lower() == 'true',
                        default=True,
                        help='Freeze Q/K projection matrices in OELM (default: True)')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum training steps')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='Warmup steps')
    parser.add_argument('--max_lr', type=float, default=5e-4,
                        help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=0.0,
                        help='Minimum learning rate')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98,
                        help='Adam beta2')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    
    # Logging arguments
    parser.add_argument('--out_dir', type=str, default='out',
                        help='Output directory')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--val_interval', type=int, default=1000,
                        help='Validation interval')
    parser.add_argument('--val_batches', type=int, default=100,
                        help='Number of validation batches')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='Checkpoint save interval')
    
    # WandB arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use WandB logging')
    parser.add_argument('--wandb_project', type=str, default='orthogonal-elm',
                        help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='WandB run name')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
