"""
Training Script for BERT Reservoir Test

Supports both Baseline (full fine-tuning) and OELM-Freeze (frozen Q/K) modes.
Implements distributed training with DDP for dual-GPU setup.

Usage:
    # Single GPU - Baseline
    python train_bert.py --freeze_mode false --lr 2e-5

    # Single GPU - OELM
    python train_bert.py --freeze_mode true --lr 1e-4

    # Dual GPU DDP - Baseline
    torchrun --nproc_per_node=2 train_bert.py --freeze_mode false

    # Dual GPU DDP - OELM
    torchrun --nproc_per_node=2 train_bert.py --freeze_mode true

    # Quick test (100 steps)
    python train_bert.py --max_steps 100 --validate_steps 50
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from tqdm import tqdm

# HuggingFace libraries
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

# Local imports
from modeling_bert_oelm import (
    load_bert_with_head_wise_orthogonal,
    print_trainable_parameters
)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_distributed() -> tuple:
    """Initialize distributed training if available."""
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
        logger.info(f"Initialized DDP: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if torch.distributed.is_initialized():
        destroy_process_group()


def load_sst2_data(
    tokenizer: BertTokenizer,
    max_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    world_size: int = 1,
    rank: int = 0
) -> tuple:
    """
    Load SST-2 dataset from GLUE benchmark.

    Args:
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size per GPU
        num_workers: DataLoader workers
        world_size: Number of GPUs (for distributed)
        rank: Current GPU rank

    Returns:
        train_loader, val_loader
    """
    logger.info("Loading SST-2 dataset from GLUE...")

    # Load from HuggingFace datasets (academic standard)
    dataset = load_dataset("glue", "sst2")

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # Tokenize datasets
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence", "idx"]
    )

    # Rename label column for compatibility
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Set format for PyTorch
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]  # 67,349 samples
    val_dataset = tokenized_datasets["validation"]  # 872 samples

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def load_mnli_data(
    tokenizer: BertTokenizer,
    max_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    world_size: int = 1,
    rank: int = 0
) -> tuple:
    """
    Load MNLI dataset from GLUE benchmark.

    Args:
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size per GPU
        num_workers: DataLoader workers
        world_size: Number of GPUs (for distributed)
        rank: Current GPU rank

    Returns:
        train_loader, val_matched_loader, val_mismatched_loader
    """
    logger.info("Loading MNLI dataset from GLUE...")

    # Load from HuggingFace datasets
    dataset = load_dataset("glue", "mnli")

    def tokenize_function(examples):
        # MNLI has premise and hypothesis
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # Tokenize datasets
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["premise", "hypothesis", "idx"]
    )

    # Rename label column for compatibility
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Set format for PyTorch
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]  # 392,702 samples
    val_matched_dataset = tokenized_datasets["validation_matched"]  # 9,815 samples
    val_mismatched_dataset = tokenized_datasets["validation_mismatched"]  # 9,832 samples

    logger.info(f"Train samples: {len(train_dataset)}, "
                f"Val matched: {len(val_matched_dataset)}, "
                f"Val mismatched: {len(val_mismatched_dataset)}")

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_matched_sampler = DistributedSampler(val_matched_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    val_mismatched_sampler = DistributedSampler(val_mismatched_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True
    )

    val_matched_loader = DataLoader(
        val_matched_dataset,
        batch_size=batch_size,
        sampler=val_matched_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    val_mismatched_loader = DataLoader(
        val_mismatched_dataset,
        batch_size=batch_size,
        sampler=val_mismatched_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_matched_loader, val_mismatched_loader


def evaluate(model, dataloader, device, is_distributed: bool = False) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Returns:
        Dict with accuracy, f1, loss
    """
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(device.type == 'cuda' and device.index != 0)):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # Get predictions
            preds = torch.argmax(logits, dim=-1)

            # Collect metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
            num_batches += 1

    # Calculate metrics (without sklearn)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy
    accuracy = np.mean(all_preds == all_labels)

    # F1 Score (binary)
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_loss = total_loss / num_batches

    model.train()

    return {
        'accuracy': accuracy,
        'f1': f1,
        'loss': avg_loss
    }


def train(args):
    """Main training function."""

    # Record start time for training duration tracking
    start_time = time.time()
    epoch_start_times = []

    # Detailed step timing statistics
    step_times = []  # Pure training time for each step (excluding validation)
    validation_times = []  # Time spent in validation
    # CRITICAL: Warmup steps - exclude from statistics (导师叮嘱 #2)
    # First 50-100 steps are slow due to CUDA context init, memory allocation,
    # and CuDNN benchmarking. We use 100 steps for safety margin.
    WARMUP_STEPS = 100  # Exclude first 100 steps from statistics

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_master = (rank == 0)

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    if is_master:
        logger.info(f"Device: {device}")
        logger.info(f"Distributed: {world_size > 1} (world_size={world_size})")
        logger.info(f"Freeze mode: {args.freeze_mode}")

    # Set random seed for reproducibility
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine learning rate
    lr = args.lr
    if lr is None:
        # Auto-select based on freeze_mode
        lr = 2e-5 if not args.freeze_mode else 1e-4

    if is_master:
        logger.info(f"Learning rate: {lr} (auto-selected: {args.lr is None})")
        logger.info(f"Mode: {'OELM-Freeze' if args.freeze_mode else 'Baseline'}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # Determine num_classes based on dataset
    if args.dataset == 'sst2':
        num_classes = 2
    elif args.dataset == 'mnli':
        num_classes = 3
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load data
    if args.dataset == 'sst2':
        train_loader, val_loader = load_sst2_data(
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            world_size=world_size,
            rank=rank
        )
    elif args.dataset == 'mnli':
        train_loader, val_loader_matched, val_loader_mismatched = load_mnli_data(
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            world_size=world_size,
            rank=rank
        )
        # For MNLI, we'll use validation_matched as primary validation set
        val_loader = val_loader_matched

    # Load model with head-wise initialization
    if is_master:
        logger.info(f"Loading BERT with head-wise {args.init_method} initialization...")

    model = load_bert_with_head_wise_orthogonal(
        model_name=args.model_name,
        num_classes=num_classes,
        freeze_mode=args.freeze_mode,
        verify_orthogonality=True,
        init_method=args.init_method
    )

    model = model.to(device)

    # Print parameter summary (master only)
    if is_master:
        print_trainable_parameters(model)

    # Wrap model with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Optimizer - only optimize parameters that require_grad
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler
    total_steps = args.max_steps if args.max_steps > 0 else len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    if is_master:
        logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        logger.info(f"Optimizer: AdamW, Scheduler: linear with warmup")

    # Training loop
    global_step = 0
    best_accuracy = 0.0
    best_step = 0

    model.train()

    # Progress bar (master only)
    pbar = tqdm(total=total_steps, desc="Training", disable=not is_master)

    for epoch in range(args.epochs):
        if is_master:
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        for batch in train_loader:
            # Check if we've reached max_steps
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            # CRITICAL: Synchronize CUDA before timing (架构师叮嘱 #1)
            if device.type == 'cuda':
                torch.cuda.synchronize()

            # Start timing this step (使用 perf_counter 获得更高精度)
            step_start_time = time.perf_counter()

            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm
                )

            optimizer.step()
            scheduler.step()

            # CRITICAL: Synchronize CUDA after computation completes (架构师叮嘱 #1)
            if device.type == 'cuda':
                torch.cuda.synchronize()

            # Record step time (exclude validation time)
            step_end_time = time.perf_counter()
            step_duration = step_end_time - step_start_time

            global_step += 1

            # Collect step time after warmup
            if global_step > WARMUP_STEPS:
                step_times.append(step_duration)

            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

            # Validation
            if global_step % args.validate_steps == 0:
                val_start_time = time.perf_counter()

                if is_master:
                    logger.info(f"\nStep {global_step}: Running validation...")

                metrics = evaluate(model, val_loader, device, world_size > 1)

                val_end_time = time.perf_counter()
                val_duration = val_end_time - val_start_time
                validation_times.append(val_duration)

                if is_master:
                    logger.info(f"Validation - Loss: {metrics['loss']:.4f}, "
                              f"Accuracy: {metrics['accuracy']:.4f}, "
                              f"F1: {metrics['f1']:.4f}")

                    # Save best model
                    if metrics['accuracy'] > best_accuracy:
                        best_accuracy = metrics['accuracy']
                        best_step = global_step

                        # Unwrap DDP if needed
                        model_to_save = model.module if isinstance(model, DDP) else model

                        save_path = Path(args.output_dir) / "best_model.pt"
                        save_path.parent.mkdir(parents=True, exist_ok=True)

                        torch.save({
                            'step': global_step,
                            'model_state_dict': model_to_save.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'accuracy': best_accuracy,
                            'args': vars(args)
                        }, save_path)

                        logger.info(f"✓ New best model saved! Accuracy: {best_accuracy:.4f}")

                model.train()

        # End of epoch
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    pbar.close()

    # Calculate total training time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    # Calculate detailed step time statistics
    if is_master and len(step_times) > 0:
        step_times_array = np.array(step_times)

        # Filter out outliers (>3x median)
        median_time = np.median(step_times_array)
        filtered_times = step_times_array[step_times_array <= 3 * median_time]

        # Calculate statistics
        mean_time = np.mean(filtered_times)
        std_time = np.std(filtered_times)
        min_time = np.min(filtered_times)
        max_time = np.max(filtered_times)
        p50_time = np.percentile(filtered_times, 50)
        p95_time = np.percentile(filtered_times, 95)
        p99_time = np.percentile(filtered_times, 99)

        # Validation time statistics
        if validation_times:
            val_times_array = np.array(validation_times)
            mean_val_time = np.mean(val_times_array)
            total_val_time = np.sum(val_times_array)
        else:
            mean_val_time = 0
            total_val_time = 0

        # Pure training time (excluding validation)
        pure_training_time = np.sum(step_times_array)

    # Final evaluation
    if is_master:
        logger.info("\n" + "="*60)
        logger.info("Training completed!")
        logger.info(f"Best accuracy: {best_accuracy:.4f} at step {best_step}")
        logger.info("="*60)

        # Log training duration
        logger.info("\n" + "="*60)
        logger.info("TRAINING TIME SUMMARY")
        logger.info("="*60)
        logger.info(f"Total wall time: {hours}h {minutes}m {seconds}s")
        logger.info(f"Total steps: {global_step}")
        logger.info(f"Mode: {'OELM-Freeze' if args.freeze_mode else 'Baseline'}")
        logger.info(f"Learning rate: {lr}")
        logger.info("="*60)

        # Detailed step timing statistics
        if len(step_times) > 0:
            logger.info("\n" + "="*60)
            logger.info("DETAILED STEP TIMING STATISTICS (excluding validation)")
            logger.info("="*60)
            logger.info(f"Steps measured: {len(step_times)} (after {WARMUP_STEPS} warmup)")
            logger.info(f"Outliers excluded: {len(step_times) - len(filtered_times)} (>3x median)")
            logger.info("")
            logger.info(f"Mean time/step: {mean_time:.4f}s")
            logger.info(f"Std dev: {std_time:.4f}s")
            logger.info(f"Min time: {min_time:.4f}s")
            logger.info(f"Max time: {max_time:.4f}s")
            logger.info(f"Median (p50): {p50_time:.4f}s")
            logger.info(f"p95: {p95_time:.4f}s")
            logger.info(f"p99: {p99_time:.4f}s")
            logger.info("")
            logger.info(f"Pure training time: {pure_training_time:.1f}s ({pure_training_time/60:.1f}min)")
            logger.info(f"Validation runs: {len(validation_times)}")
            logger.info(f"Total validation time: {total_val_time:.1f}s ({total_val_time/60:.1f}min)")
            logger.info(f"Avg validation time: {mean_val_time:.2f}s")
            logger.info("="*60)

            # Save timing data to file for analysis
            timing_data = {
                'mode': 'oelm' if args.freeze_mode else 'baseline',
                'total_wall_time': total_time,
                'pure_training_time': float(pure_training_time),
                'total_validation_time': float(total_val_time),
                'num_steps_measured': len(step_times),
                'mean_time_per_step': float(mean_time),
                'std_time_per_step': float(std_time),
                'min_time': float(min_time),
                'max_time': float(max_time),
                'median_time': float(p50_time),
                'p95_time': float(p95_time),
                'p99_time': float(p99_time),
                'step_times': step_times,
                'validation_times': validation_times
            }

            import json
            timing_file = Path(args.output_dir) / 'timing_stats.json'
            timing_file.parent.mkdir(parents=True, exist_ok=True)
            with open(timing_file, 'w') as f:
                json.dump(timing_data, f, indent=2)
            logger.info(f"Timing data saved to: {timing_file}")

        # Check if target was met
        if best_accuracy >= 0.80:
            logger.info(f"✓ TARGET MET! Accuracy {best_accuracy:.4f} >= 80%")
        else:
            logger.info(f"✗ Target not met. Accuracy {best_accuracy:.4f} < 80%")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="BERT Reservoir Test Training")

    # Model arguments
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='BERT model name')
    parser.add_argument('--freeze_mode', type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=False,
                        help='True for OELM (freeze Q/K), False for Baseline')
    parser.add_argument('--init_method', type=str, default='orthogonal',
                        choices=['orthogonal', 'normal'],
                        help='Q/K initialization method: orthogonal (QR) or normal (Gaussian)')

    # Training arguments
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (None=auto: 2e-5 for baseline, 1e-4 for OELM)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='Max training steps (-1 = use epochs)')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max sequence length')

    # Optimization arguments
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio (0.1 = 10% of total steps)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (-1 to disable)')

    # Validation arguments
    parser.add_argument('--validate_steps', type=int, default=500,
                        help='Validate every N steps')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='sst2',
                        choices=['sst2', 'mnli'],
                        help='Dataset: sst2 (2-class) or mnli (3-class NLI)')

    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
