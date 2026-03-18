#!/usr/bin/env python3
"""
OELM Pretraining Script
Supports QK freezing and FFN freezing during pretraining
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import GPT2Tokenizer, GPT2Config, get_cosine_schedule_with_warmup
from datasets import load_dataset
import argparse
import os
import json
from tqdm import tqdm
import logging
from datetime import datetime


# Setup logging
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(
        output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


class OELMGPT2(nn.Module):
    """GPT-2 with optional QK and FFN freezing"""

    def __init__(self, config, freeze_qk=False, freeze_ffn=False):
        super().__init__()
        from transformers import GPT2LMHeadModel

        self.model = GPT2LMHeadModel(config)
        self.freeze_qk = freeze_qk
        self.freeze_ffn = freeze_ffn

        self._apply_freezing()

    def _apply_freezing(self):
        """Apply parameter freezing"""
        frozen_params = 0
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += param.numel()

            # Freeze Q/K projections
            if self.freeze_qk and any(
                x in name.lower()
                for x in ["q_proj", "k_proj", "query", "key", "c_attn"]
            ):
                if "q_proj" in name or "query" in name or name.endswith(".q.weight"):
                    param.requires_grad = False
                    frozen_params += param.numel()
                    logging.info(f"Frozen Q: {name}")
                elif "k_proj" in name or "key" in name or name.endswith(".k.weight"):
                    param.requires_grad = False
                    frozen_params += param.numel()
                    logging.info(f"Frozen K: {name}")

            # Freeze FFN layers
            if self.freeze_ffn and any(
                x in name.lower()
                for x in ["mlp", "ffn", "up_proj", "down_proj", "c_fc", "c_proj"]
            ):
                param.requires_grad = False
                frozen_params += param.numel()
                logging.info(f"Frozen FFN: {name}")

        trainable = total_params - frozen_params
        logging.info(f"Total params: {total_params:,}")
        logging.info(
            f"Frozen params: {frozen_params:,} ({frozen_params / total_params * 100:.1f}%)"
        )
        logging.info(
            f"Trainable params: {trainable:,} ({trainable / total_params * 100:.1f}%)"
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)


def get_dataloader(dataset_name, tokenizer, batch_size, seq_length, split="train"):
    """Load and prepare dataset"""

    if dataset_name == "tinystories":
        dataset = load_dataset("roneneldan/TinyStories", split=split)
        text_column = "text"
    elif dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext", split=split)
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def tokenize_function(examples):
        # Tokenize and create chunks
        tokens = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=seq_length,
            return_overflowing_tokens=True,
        )
        return tokens

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    def group_texts(examples):
        # Concatenate all texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])

        # Drop remainder
        total_length = (total_length // seq_length) * seq_length

        # Split by chunks
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
    )

    return DataLoader(
        lm_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
    )


def train(args):
    """Main training function"""

    # Initialize distributed training
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        is_distributed = True
    else:
        local_rank = 0
        is_distributed = False

    # Setup logging
    if local_rank == 0:
        logger = setup_logging(args.output_dir)
    else:
        logger = logging.getLogger(__name__)

    logger.info(f"Starting training with args: {args}")
    logger.info(f"Distributed: {is_distributed}, Local rank: {local_rank}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    config = GPT2Config.from_pretrained(args.model_name)
    model = OELMGPT2(config, freeze_qk=args.freeze_qk, freeze_ffn=args.freeze_ffn)

    device = torch.device(f"cuda:{local_rank}")
    model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Load data
    train_loader = get_dataloader(
        args.dataset, tokenizer, args.batch_size, args.seq_length, "train"
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Setup scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    # Training loop
    global_step = 0
    best_ppl = float("inf")

    model.train()
    optimizer.zero_grad()

    progress_bar = tqdm(total=args.max_steps, disable=local_rank != 0)

    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            input_ids = torch.stack([torch.tensor(x) for x in batch["input_ids"]]).to(
                device
            )
            attention_mask = torch.stack(
                [torch.tensor(x) for x in batch["attention_mask"]]
            ).to(device)
            labels = torch.stack([torch.tensor(x) for x in batch["labels"]]).to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if global_step % args.logging_steps == 0 and local_rank == 0:
                lr = scheduler.get_last_lr()[0]
                ppl = torch.exp(loss * args.gradient_accumulation_steps).item()
                logger.info(
                    f"Step {global_step}: loss={loss.item():.4f}, ppl={ppl:.2f}, lr={lr:.2e}"
                )
                progress_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "ppl": f"{ppl:.2f}"}
                )

            # Save checkpoint
            if global_step % args.save_steps == 0 and local_rank == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                model_to_save = model.module if is_distributed else model
                model_to_save.model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

                # Save training state
                state = {
                    "step": global_step,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "loss": loss.item(),
                }
                torch.save(state, os.path.join(save_path, "training_state.pt"))

                logger.info(f"Saved checkpoint to {save_path}")

            global_step += 1
            progress_bar.update(1)

    progress_bar.close()

    # Save final model
    if local_rank == 0:
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        model_to_save = model.module if is_distributed else model
        model_to_save.model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Training completed! Final model saved to {final_path}")

    if is_distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="OELM Pretraining")

    # Data arguments
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["tinystories", "openwebtext"]
    )
    parser.add_argument("--model_name", type=str, default="gpt2")

    # Training arguments
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine"]
    )
    parser.add_argument("--warmup_steps", type=int, default=1000)

    # OELM arguments
    parser.add_argument(
        "--freeze_qk", type=str, default="false", choices=["true", "false"]
    )
    parser.add_argument(
        "--freeze_ffn", type=str, default="false", choices=["true", "false"]
    )

    # Logging arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)

    # Other arguments
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--ddp_find_unused_parameters", type=str, default="false")

    args = parser.parse_args()

    # Convert string to bool
    args.freeze_qk = args.freeze_qk.lower() == "true"
    args.freeze_ffn = args.freeze_ffn.lower() == "true"

    train(args)


if __name__ == "__main__":
    main()
