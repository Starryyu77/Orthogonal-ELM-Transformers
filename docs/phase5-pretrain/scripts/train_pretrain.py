"""
OELM Pre-training Script

支持从零开始预训练OELM模型，对比:
- Baseline: 标准GPT预训练
- OELM-QK: 冻结Q/K
- OELM-QK-FFN: 冻结Q/K+FFN

数据集: TinyStories, OpenWebText
"""

import os
import sys
import json
import time
import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineLRScheduler

from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
from modeling_oelm_pretrain import create_model, OELMConfig


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""

    def __init__(
        self,
        texts: list,
        tokenizer: GPT2Tokenizer,
        max_length: int = 1024,
        stride: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = []

        print(f"Tokenizing {len(texts)} texts...")
        for text in tqdm(texts, desc="Processing"):
            # Tokenize
            tokenized = tokenizer(
                text, max_length=max_length, truncation=True, return_tensors="pt"
            )

            input_ids = tokenized["input_ids"].squeeze(0)

            # Create overlapping windows for longer texts
            if len(input_ids) > max_length:
                for i in range(0, len(input_ids) - max_length + 1, stride):
                    chunk = input_ids[i : i + max_length]
                    self.examples.append(chunk)
            else:
                self.examples.append(input_ids)

        print(f"Created {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_data(
    dataset_name: str,
    tokenizer: GPT2Tokenizer,
    max_length: int = 512,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    """Load and prepare dataset"""

    print(f"\nLoading dataset: {dataset_name}")

    if dataset_name == "tinystories":
        dataset = load_dataset(
            "roneneldan/TinyStories", split="train", cache_dir=cache_dir
        )
        texts = [item["text"] for item in dataset]
    elif dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext", split="train", cache_dir=cache_dir)
        texts = [item["text"] for item in dataset]
    elif dataset_name == "wikitext103":
        dataset = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="train", cache_dir=cache_dir
        )
        texts = [item["text"] for item in dataset if item["text"].strip()]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Limit samples if specified
    if max_samples:
        texts = texts[:max_samples]

    print(f"Loaded {len(texts)} texts")

    # Create dataset
    train_dataset = TextDataset(texts, tokenizer, max_length=max_length)

    return train_dataset


def get_lr_scheduler(optimizer, warmup_steps, total_steps, lr, min_lr=1e-7):
    """Create cosine learning rate scheduler"""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_perplexity(loss):
    """Compute perplexity from loss"""
    return math.exp(loss)


def save_checkpoint(
    model, optimizer, scheduler, step, loss, output_dir, method, is_best=False
):
    """Save training checkpoint"""
    checkpoint_dir = output_dir / method / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "config": model.config.__dict__,
        },
        checkpoint_dir / "pytorch_model.pt",
    )

    # Save best model
    if is_best:
        best_dir = output_dir / method / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "loss": loss,
                "config": model.config.__dict__,
            },
            best_dir / "pytorch_model.pt",
        )

    print(f"Saved checkpoint at step {step}")


def train(args):
    """Main training function"""

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    print(f"\nCreating model: {args.method}")
    model = create_model(
        model_size=args.model_size, method=args.method, vocab_size=tokenizer.vocab_size
    )
    model = model.to(device)

    # Load data
    train_dataset = load_data(
        args.dataset,
        tokenizer,
        max_length=args.max_seq_len,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
    )

    # Collate function for padding variable-length sequences
    def collate_fn(batch):
        max_len = max(len(x) for x in batch)
        padded_batch = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, x in enumerate(batch):
            padded_batch[i, :len(x)] = x
        return padded_batch

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    def collate_fn(batch):
        max_len = max(len(x) for x in batch)
        padded_batch = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, x in enumerate(batch):
            padded_batch[i, :len(x)] = x
        return padded_batch
    def collate_fn(batch):
        max_len = max(len(x) for x in batch)
        padded = []
        for x in batch:
            if len(x) < max_len:
                padding = torch.zeros(max_len - len(x), dtype=torch.long)
                x = torch.cat([x, padding])
            padded.append(x)
        return torch.stack(padded)

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Setup optimizer
    # Different learning rates for different methods
    if args.method == "baseline":
        lr = args.lr if args.lr else 3e-4
    else:
        lr = args.lr if args.lr else 1e-3  # Higher LR for OELM

    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    # Calculate total steps
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = num_update_steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    print(f"\nTraining setup:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")

    # Create scheduler
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, lr)

    # Resume from checkpoint if specified
    start_step = 0
    best_loss = float("inf")

    if args.resume_from:
        print(f"\nResuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"]
        best_loss = checkpoint.get("loss", float("inf"))

    # Training loop
    print(f"\n{'=' * 60}")
    print("Starting training...")
    print("=" * 60)

    model.train()
    global_step = start_step
    running_loss = 0.0
    log_interval = 100
    save_interval = args.save_steps

    start_time = time.time()

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=args.disable_tqdm,
        )

        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            batch = batch.to(device)

            # Forward pass
            _, loss = model(batch, labels=batch)

            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            running_loss += loss.item()
            epoch_loss += loss.item()

            # Update weights
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_steps += 1

                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = (
                    running_loss * args.gradient_accumulation_steps / log_interval
                )

                progress_bar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "ppl": f"{compute_perplexity(avg_loss):.2f}",
                        "lr": f"{current_lr:.2e}",
                    }
                )

                # Logging
                if global_step % log_interval == 0:
                    running_loss = 0.0

                # Save checkpoint
                if global_step % save_interval == 0:
                    avg_loss = epoch_loss / epoch_steps
                    is_best = avg_loss < best_loss
                    if is_best:
                        best_loss = avg_loss

                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        avg_loss,
                        output_dir,
                        args.method,
                        is_best,
                    )

        # End of epoch
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        epoch_ppl = compute_perplexity(avg_epoch_loss)

        print(f"\nEpoch {epoch + 1} completed:")
        print(f"  Average loss: {avg_epoch_loss:.4f}")
        print(f"  Perplexity: {epoch_ppl:.2f}")
        print(f"  Learning rate: {current_lr:.2e}")

        # Save at end of each epoch
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            global_step,
            avg_epoch_loss,
            output_dir,
            args.method,
            is_best=(avg_epoch_loss < best_loss),
        )

    # Training completed
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print("=" * 60)
    print(f"Total time: {hours}h {minutes}m")
    print(f"Final loss: {avg_epoch_loss:.4f}")
    print(f"Final perplexity: {epoch_ppl:.2f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir / args.method}")


def main():
    parser = argparse.ArgumentParser(description="OELM Pre-training")

    # Model config
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "medium"],
        help="Model size",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="baseline",
        choices=["baseline", "oelm_qk", "oelm_qk_ffn"],
        help="Training method",
    )

    # Data config
    parser.add_argument(
        "--dataset",
        type=str,
        default="tinystories",
        choices=["tinystories", "openwebtext", "wikitext103"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples (for testing)",
    )

    # Training config
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 3e-4 for baseline, 1e-3 for OELM)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save checkpoint every N steps"
    )

    # System config
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Dataset cache directory"
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--disable_tqdm", action="store_true", help="Disable tqdm progress bar"
    )

    args = parser.parse_args()

    # Print config
    print(f"\n{'=' * 60}")
    print("OELM Pre-training Configuration")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Train
    train(args)


if __name__ == "__main__":
    main()
