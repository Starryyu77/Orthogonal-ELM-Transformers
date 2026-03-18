#!/usr/bin/env python3
"""
OELM Pretraining Script - Mini Model Version (30-60M params)
Optimized for TinyStories dataset
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


class OELMGPT2Mini(nn.Module):
    """GPT-2 Mini (30-60M params) with optional QK and FFN freezing"""

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

            # Freeze Q/K projections (attention layers)
            if self.freeze_qk:
                # GPT-2 uses Conv1D for attention, look for c_attn weight splits
                if "attn.c_attn" in name:
                    # c_attn contains q, k, v projections concatenated
                    # We need to freeze q and k portions
                    # For Conv1D: weight shape is [3*n_embd, n_embd]
                    # First third is q, second is k, third is v
                    if "weight" in name:
                        n_embd = param.shape[1]  # input dimension
                        # Create a mask: True for trainable, False for frozen
                        mask = torch.ones_like(param, dtype=torch.bool)
                        # Freeze first 2/3 (q and k), keep last 1/3 (v) trainable
                        mask[: 2 * n_embd, :] = False
                        param.register_hook(lambda grad, mask=mask: grad * mask.float())
                        # Mark as frozen for counting
                        frozen_count = (2 * n_embd) * n_embd
                        frozen_params += frozen_count
                        logging.info(f"Frozen Q,K in {name}: {frozen_count:,} params")
                elif any(x in name for x in ["q_proj", "k_proj"]):
                    param.requires_grad = False
                    frozen_params += param.numel()
                    logging.info(f"Frozen: {name}")

            # Freeze FFN layers
            if self.freeze_ffn:
                if any(x in name.lower() for x in ["mlp", "ffn", "c_fc", "c_proj"]):
                    param.requires_grad = False
                    frozen_params += param.numel()
                    logging.info(f"Frozen FFN: {name}")

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        actual_frozen = total_params - trainable
        logging.info(f"=" * 60)
        logging.info(f"Total params: {total_params:,}")
        logging.info(
            f"Frozen params: {actual_frozen:,} ({actual_frozen / total_params * 100:.1f}%)"
        )
        logging.info(
            f"Trainable params: {trainable:,} ({trainable / total_params * 100:.1f}%)"
        )
        logging.info(f"=" * 60)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)


def get_dataloader(
    dataset_name, tokenizer, batch_size, seq_length, split="train", streaming=False
):
    """Load and prepare dataset"""

    if dataset_name == "tinystories":
        dataset = load_dataset(
            "roneneldan/TinyStories", split=split, streaming=streaming
        )
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=seq_length,
            padding="max_length",
            return_tensors=None,
        )

    if streaming:
        tokenized = dataset.map(
            tokenize_function, batched=True, remove_columns=[text_column]
        )
        # For streaming, we need to batch manually
        return tokenized
    else:
        tokenized = dataset.map(
            tokenize_function, batched=True, remove_columns=[text_column], num_proc=4
        )

        return DataLoader(
            tokenized,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
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
        logger.addHandler(logging.NullHandler())

    logger.info(f"Starting training with args: {args}")
    logger.info(f"Distributed: {is_distributed}, Local rank: {local_rank}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create mini config
    # GPT-2 small: 124M params (d=768, l=12, h=12)
    # GPT-2 mini: ~42M params (d=512, l=6, h=8)
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=512,  # reduced from 768
        n_layer=6,  # reduced from 12
        n_head=8,  # reduced from 12
        n_inner=2048,  # FFN dim = 4*n_embd
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
    )

    logger.info(
        f"Model config: n_embd={config.n_embd}, n_layer={config.n_layer}, n_head={config.n_head}"
    )

    # Load model
    model = OELMGPT2Mini(config, freeze_qk=args.freeze_qk, freeze_ffn=args.freeze_ffn)

    device = torch.device(f"cuda:{local_rank}")
    model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Load data
    logger.info(f"Loading dataset: {args.dataset}")
    train_loader = get_dataloader(
        args.dataset, tokenizer, args.batch_size, args.seq_length, "train"
    )

    # Setup optimizer - only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )

    # Setup scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    # Training loop
    global_step = 0
    epoch = 0

    model.train()
    optimizer.zero_grad()

    if local_rank == 0:
        progress_bar = tqdm(total=args.max_steps, desc="Training")

    while global_step < args.max_steps:
        epoch += 1
        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= args.max_steps:
                break

            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).to(device)
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).to(
                device
            )
            labels = input_ids.clone()

            # Forward
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward
            loss.backward()

            # Gradient clipping and update
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            loss_item = loss.item()
            epoch_loss += loss_item
            num_batches += 1

            if global_step % args.logging_steps == 0 and local_rank == 0:
                lr = scheduler.get_last_lr()[0]
                ppl = torch.exp(torch.tensor(loss_item)).item()
                logger.info(
                    f"Step {global_step} | Loss: {loss_item:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}"
                )
                if "progress_bar" in locals():
                    progress_bar.set_postfix(
                        {"loss": f"{loss_item:.4f}", "ppl": f"{ppl:.2f}"}
                    )

            # Save checkpoint
            if global_step % args.save_steps == 0 and local_rank == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                model_to_save = model.module if is_distributed else model
                model_to_save.model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

                # Save config
                with open(os.path.join(save_path, "model_config.json"), "w") as f:
                    json.dump(
                        {
                            "n_embd": config.n_embd,
                            "n_layer": config.n_layer,
                            "n_head": config.n_head,
                            "freeze_qk": args.freeze_qk,
                            "freeze_ffn": args.freeze_ffn,
                        },
                        f,
                        indent=2,
                    )

                logger.info(f"Saved checkpoint to {save_path}")

            global_step += 1
            if local_rank == 0 and "progress_bar" in locals():
                progress_bar.update(1)

        # Epoch summary
        if local_rank == 0 and num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
            logger.info(
                f"Epoch {epoch} completed | Avg Loss: {avg_loss:.4f} | Avg PPL: {avg_ppl:.2f}"
            )

    if local_rank == 0 and "progress_bar" in locals():
        progress_bar.close()

    # Save final model
    if local_rank == 0:
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        model_to_save = model.module if is_distributed else model
        model_to_save.model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)

        # Save final config
        with open(os.path.join(final_path, "model_config.json"), "w") as f:
            json.dump(
                {
                    "n_embd": config.n_embd,
                    "n_layer": config.n_layer,
                    "n_head": config.n_head,
                    "freeze_qk": args.freeze_qk,
                    "freeze_ffn": args.freeze_ffn,
                    "total_steps": global_step,
                    "final_loss": loss_item,
                },
                f,
                indent=2,
            )

        logger.info(f"=" * 60)
        logger.info(f"Training completed!")
        logger.info(f"Final model saved to {final_path}")
        logger.info(f"Total steps: {global_step}")
        logger.info(f"=" * 60)

    if is_distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="OELM Pretraining - Mini Model for TinyStories"
    )

    # Data arguments
    parser.add_argument(
        "--dataset", type=str, default="tinystories", choices=["tinystories"]
    )

    # Model arguments
    parser.add_argument("--model_name", type=str, default="gpt2")  # Only for tokenizer

    # Training arguments
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    # OELM arguments
    parser.add_argument(
        "--freeze_qk", action="store_true", help="Freeze Q/K projections"
    )
    parser.add_argument("--freeze_ffn", action="store_true", help="Freeze FFN layers")

    # Logging arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=100)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
