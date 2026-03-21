#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset, load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup

from common import (
    base_manifest,
    base_resource_usage,
    build_run_dir,
    build_run_name,
    ensure_dir,
    finalize_resource_usage,
    peak_gpu_memory_gb,
    set_seed,
    setup_logging,
    write_json,
)
from modeling import OELMForLanguageModeling, create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OELM pretraining V2")
    parser.add_argument("--phase", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["tinystories", "openwebtext"], required=True)
    parser.add_argument("--method", type=str, choices=["baseline", "qk_only", "qk_ffn"], required=True)
    parser.add_argument("--model_size", type=str, choices=["mini", "small"], required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--preprocessing_num_proc", type=int, default=1)
    parser.add_argument("--validation_split_pct", type=float, default=0.01)
    parser.add_argument("--max_train_documents", type=int, default=None)
    parser.add_argument("--max_eval_documents", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=50)
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


def init_distributed() -> tuple[bool, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 1
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return True, local_rank, world_size


def is_rank_zero(is_distributed: bool) -> bool:
    return (not is_distributed) or dist.get_rank() == 0


def category_for_tensor(name: str) -> str:
    if name.startswith("token_embedding") or name.startswith("position_embedding"):
        return "embeddings"
    if ".attn.W_q." in name:
        return "attn_q"
    if ".attn.W_k." in name:
        return "attn_k"
    if ".attn.W_v." in name:
        return "attn_v"
    if ".attn.W_o." in name:
        return "attn_out"
    if ".ffn.up_proj." in name:
        return "mlp_in"
    if ".ffn.down_proj." in name:
        return "mlp_out"
    if ".ln" in name or name.startswith("ln_f"):
        return "ln"
    if name.startswith("lm_head"):
        return "lm_head"
    return "other"


def build_freeze_audit(
    model: OELMForLanguageModeling,
    optimizer_param_count: int,
    method: str,
    phase: str,
    seed: int,
) -> dict[str, Any]:
    seen: set[int] = set()
    entries: list[dict[str, Any]] = []
    category_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"trainable": 0, "frozen": 0, "total": 0}
    )

    def consume(named_tensors, source: str) -> None:
        for name, tensor in named_tensors:
            if tensor is None or id(tensor) in seen:
                continue
            seen.add(id(tensor))
            trainable = bool(getattr(tensor, "requires_grad", False))
            category = category_for_tensor(name)
            numel = int(tensor.numel())
            category_totals[category]["total"] += numel
            if trainable:
                category_totals[category]["trainable"] += numel
            else:
                category_totals[category]["frozen"] += numel
            entries.append(
                {
                    "name": name,
                    "category": category,
                    "source": source,
                    "shape": list(tensor.shape),
                    "numel": numel,
                    "trainable": trainable,
                }
            )

    consume(model.named_parameters(), "parameter")
    consume(model.named_buffers(), "buffer")

    total_params = sum(item["numel"] for item in entries)
    trainable_params = sum(item["numel"] for item in entries if item["trainable"])
    frozen_params = total_params - trainable_params
    return {
        "phase": phase,
        "method": method,
        "seed": seed,
        "totals": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_ratio": trainable_params / total_params if total_params else 0.0,
            "frozen_ratio": frozen_params / total_params if total_params else 0.0,
            "optimizer_param_count": optimizer_param_count,
            "optimizer_matches_trainable_audit": optimizer_param_count == trainable_params,
        },
        "category_totals": dict(category_totals),
        "entries": entries,
    }


def prepare_openwebtext_splits(args: argparse.Namespace):
    dataset = load_dataset("openwebtext", split="train", cache_dir=args.cache_dir)
    dataset = dataset.shuffle(seed=args.seed)

    if args.max_train_documents is not None or args.max_eval_documents is not None:
        requested_train = args.max_train_documents or len(dataset)
        requested_eval = args.max_eval_documents or max(1, int(requested_train * args.validation_split_pct))
        total = min(len(dataset), requested_train + requested_eval)
        dataset = dataset.select(range(total))
        train_size = min(requested_train, len(dataset) - requested_eval)
        eval_size = min(requested_eval, len(dataset) - train_size)
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, train_size + eval_size))
        return train_dataset, eval_dataset, "text"

    split = dataset.train_test_split(test_size=args.validation_split_pct, seed=args.seed)
    return split["train"], split["test"], "text"


def load_lm_splits(args: argparse.Namespace):
    if args.dataset == "tinystories":
        train_dataset = load_dataset(
            "roneneldan/TinyStories",
            split="train",
            cache_dir=args.cache_dir,
        )
        eval_dataset = load_dataset(
            "roneneldan/TinyStories",
            split="validation",
            cache_dir=args.cache_dir,
        )
        return train_dataset, eval_dataset, "text"
    return prepare_openwebtext_splits(args)


def tokenize_lm_dataset(
    dataset: Dataset,
    *,
    tokenizer: GPT2Tokenizer,
    text_key: str,
    seq_length: int,
    num_proc: int,
) -> Dataset:
    def tokenize_function(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch[text_key],
            truncation=False,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing documents",
    )

    def group_texts(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = {key: sum(batch[key], []) for key in batch.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // seq_length) * seq_length
        input_ids = [
            concatenated["input_ids"][index : index + seq_length]
            for index in range(0, total_length, seq_length)
        ]
        attention_mask = [[1] * seq_length for _ in input_ids]
        labels = [chunk[:] for chunk in input_ids]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    grouped = tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Grouping tokens into fixed-length chunks",
    )
    grouped.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return grouped


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    is_distributed: bool,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


@torch.no_grad()
def evaluate_lm(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += float(outputs["loss"].item())
        total_batches += 1
        if total_batches >= max_batches:
            break
    model.train()
    avg_loss = total_loss / max(total_batches, 1)
    return {
        "loss": avg_loss,
        "ppl": math.exp(avg_loss) if avg_loss < 20 else float("inf"),
        "batches": total_batches,
    }


def save_checkpoint(
    checkpoint_dir: Path,
    model: OELMForLanguageModeling,
    tokenizer: GPT2Tokenizer,
    *,
    extra_metadata: dict[str, Any],
) -> None:
    model.save_checkpoint(checkpoint_dir, tokenizer, extra_metadata=extra_metadata)


def train(args: argparse.Namespace) -> None:
    is_distributed, local_rank, world_size = init_distributed()
    rank0 = is_rank_zero(is_distributed)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + local_rank)

    run_name = build_run_name(args.run_name)
    output_dir = build_run_dir(args.output_root, args.phase, args.method, args.seed, run_name)
    logger = None
    if rank0:
        logger, log_path = setup_logging(output_dir, prefix=f"{args.phase}_{args.method}")

    start_time_utc = datetime.now(tz=timezone.utc)
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = create_model(
        model_size=args.model_size,
        method=args.method,
        vocab_size=len(tokenizer),
        max_seq_len=args.seq_length,
    )
    model.to(device)
    train_dataset_raw, eval_dataset_raw, text_key = load_lm_splits(args)
    train_dataset = tokenize_lm_dataset(
        train_dataset_raw,
        tokenizer=tokenizer,
        text_key=text_key,
        seq_length=args.seq_length,
        num_proc=args.preprocessing_num_proc,
    )
    eval_dataset = tokenize_lm_dataset(
        eval_dataset_raw,
        tokenizer=tokenizer,
        text_key=text_key,
        seq_length=args.seq_length,
        num_proc=args.preprocessing_num_proc,
    )

    train_loader, train_sampler = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_distributed=is_distributed,
        shuffle=True,
    )
    eval_loader, _ = build_dataloader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_distributed=False,
        shuffle=False,
    )

    optimizer_params = [param for param in model.parameters() if param.requires_grad]
    optimizer_param_count = sum(param.numel() for param in optimizer_params)
    freeze_audit = build_freeze_audit(
        model=model,
        optimizer_param_count=optimizer_param_count,
        method=args.method,
        phase=args.phase,
        seed=args.seed,
    )
    if rank0:
        write_json(output_dir / "freeze_audit.json", freeze_audit)

    manifest = base_manifest(
        phase=args.phase,
        method=args.method,
        seed=args.seed,
        dataset=args.dataset,
        output_dir=output_dir,
        extra={
            "model_size": args.model_size,
            "run_name": run_name,
            "seq_length": args.seq_length,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "log_file": str(log_path) if rank0 else None,
        },
    )
    resource_usage = base_resource_usage(
        phase=args.phase,
        method=args.method,
        seed=args.seed,
        dataset=args.dataset,
        start_time_utc=start_time_utc,
        extra={
            "model_size": args.model_size,
            "output_dir": str(output_dir),
            "run_name": run_name,
        },
    )
    if rank0:
        write_json(output_dir / "run_manifest.json", manifest)

    if rank0 and logger is not None:
        logger.info("Output directory: %s", output_dir)
        logger.info("Tokenizer vocab size: %s", len(tokenizer))
        logger.info("Train chunks: %s | Eval chunks: %s", len(train_dataset), len(eval_dataset))
        logger.info("Freeze audit optimizer match: %s", freeze_audit["totals"]["optimizer_matches_trainable_audit"])

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    progress_bar = tqdm(total=args.max_steps, disable=not rank0, desc=f"{args.phase}:{args.method}")
    global_step = 0
    micro_step = 0
    epoch = 0
    best_eval_loss = float("inf")
    best_step = -1
    tokens_seen_local = 0
    history: list[dict[str, Any]] = []
    last_loss = None
    model.train()
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_steps:
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            raw_loss = outputs["loss"]
            loss = raw_loss / args.gradient_accumulation_steps
            loss.backward()
            tokens_seen_local += int(input_ids.numel())
            last_loss = float(raw_loss.item())
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if rank0:
                progress_bar.update(1)

            if rank0 and logger is not None and global_step % args.logging_steps == 0:
                logger.info(
                    "Step %s | loss=%.4f | ppl=%.4f | lr=%.2e",
                    global_step,
                    last_loss,
                    math.exp(last_loss) if last_loss is not None and last_loss < 20 else float("inf"),
                    scheduler.get_last_lr()[0],
                )

            if global_step % args.eval_steps == 0:
                if is_distributed:
                    dist.barrier()
                if rank0:
                    metrics = evaluate_lm(
                        model.module if is_distributed else model,
                        eval_loader,
                        device,
                        args.max_eval_batches,
                    )
                    history.append(
                        {
                            "step": global_step,
                            "train_loss": last_loss,
                            "eval_loss": metrics["loss"],
                            "eval_ppl": metrics["ppl"],
                            "eval_batches": metrics["batches"],
                        }
                    )
                    if logger is not None:
                        logger.info(
                            "Eval step %s | eval_loss=%.4f | eval_ppl=%.4f",
                            global_step,
                            metrics["loss"],
                            metrics["ppl"],
                        )
                    checkpoint_payload = {
                        "phase": args.phase,
                        "dataset": args.dataset,
                        "method": args.method,
                        "seed": args.seed,
                        "step": global_step,
                        "eval_loss": metrics["loss"],
                        "eval_ppl": metrics["ppl"],
                    }
                    save_checkpoint(
                        output_dir / f"checkpoint-{global_step}",
                        model.module if is_distributed else model,
                        tokenizer,
                        extra_metadata=checkpoint_payload,
                    )
                    if metrics["loss"] < best_eval_loss:
                        best_eval_loss = metrics["loss"]
                        best_step = global_step
                        save_checkpoint(
                            output_dir / "best",
                            model.module if is_distributed else model,
                            tokenizer,
                            extra_metadata=checkpoint_payload,
                        )
                if is_distributed:
                    dist.barrier()

            if global_step >= args.max_steps:
                break

    if rank0:
        progress_bar.close()

    if is_distributed:
        token_tensor = torch.tensor([tokens_seen_local], device=device, dtype=torch.long)
        dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
        tokens_seen = int(token_tensor.item())
    else:
        tokens_seen = tokens_seen_local

    if rank0:
        final_model = model.module if is_distributed else model
        final_eval = evaluate_lm(final_model, eval_loader, device, args.max_eval_batches)
        save_checkpoint(
            output_dir / "final",
            final_model,
            tokenizer,
            extra_metadata={
                "phase": args.phase,
                "dataset": args.dataset,
                "method": args.method,
                "seed": args.seed,
                "step": global_step,
                "eval_loss": final_eval["loss"],
                "eval_ppl": final_eval["ppl"],
            },
        )
        training_summary = {
            "phase": args.phase,
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "model_size": args.model_size,
            "global_step": global_step,
            "last_train_loss": last_loss,
            "final_eval_loss": final_eval["loss"],
            "final_eval_ppl": final_eval["ppl"],
            "best_eval_loss": best_eval_loss if best_step > 0 else final_eval["loss"],
            "best_step": best_step if best_step > 0 else global_step,
            "history": history,
        }
        write_json(output_dir / "training_summary.json", training_summary)
        end_time_utc = datetime.now(tz=timezone.utc)
        finalize_resource_usage(
            resource_usage,
            end_time_utc=end_time_utc,
            state="completed",
            steps=global_step,
            tokens_seen=tokens_seen,
            peak_gpu_mem_gb=peak_gpu_memory_gb(),
            extra={
                "best_eval_loss": training_summary["best_eval_loss"],
                "final_eval_loss": training_summary["final_eval_loss"],
            },
        )
        write_json(output_dir / "resource_usage.json", resource_usage)
        manifest["completed_at_utc"] = end_time_utc.isoformat(timespec="seconds")
        manifest["state"] = "completed"
        write_json(output_dir / "run_manifest.json", manifest)
        if logger is not None:
            logger.info("Training completed at %s", end_time_utc.isoformat(timespec="seconds"))
            logger.info("Best eval loss: %.4f at step %s", training_summary["best_eval_loss"], training_summary["best_step"])

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
