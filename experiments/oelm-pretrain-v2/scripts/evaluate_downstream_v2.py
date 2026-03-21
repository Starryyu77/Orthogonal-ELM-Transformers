#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup

from common import (
    base_manifest,
    base_resource_usage,
    build_run_dir,
    build_run_name,
    load_config,
    peak_gpu_memory_gb,
    set_seed,
    setup_logging,
    write_json,
)
from modeling import OELMForLanguageModeling, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OELM downstream evaluation V2")
    parser.add_argument("--phase", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["baseline", "qk_only", "qk_ffn"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, choices=["last_non_pad", "mean"], default="last_non_pad")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--eval_on_test", action="store_true")
    return parser.parse_args()


def compute_accuracy(labels: list[int], preds: list[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(int(label == pred) for label, pred in zip(labels, preds))
    return correct / len(labels)


def compute_weighted_f1(labels: list[int], preds: list[int]) -> float:
    if not labels:
        return 0.0
    classes = sorted(set(labels) | set(preds))
    total_support = len(labels)
    weighted_sum = 0.0
    for cls in classes:
        tp = sum(1 for label, pred in zip(labels, preds) if label == cls and pred == cls)
        fp = sum(1 for label, pred in zip(labels, preds) if label != cls and pred == cls)
        fn = sum(1 for label, pred in zip(labels, preds) if label == cls and pred != cls)
        support = sum(1 for label in labels if label == cls)
        if support == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        weighted_sum += f1 * support
    return weighted_sum / total_support


def compute_binary_f1(labels: list[int], preds: list[int], positive_label: int = 1) -> float:
    tp = sum(1 for label, pred in zip(labels, preds) if label == positive_label and pred == positive_label)
    fp = sum(1 for label, pred in zip(labels, preds) if label != positive_label and pred == positive_label)
    fn = sum(1 for label, pred in zip(labels, preds) if label == positive_label and pred != positive_label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_task_spec(config_path: str | Path, task_name: str, setting_name: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    config = load_config(config_path)
    task_spec = config["tasks"][task_name]
    settings = config.get("settings", {})
    if setting_name not in settings:
        raise ValueError(f"Setting {setting_name} not found in {config_path}")
    return task_spec, settings[setting_name], config.get("seeds", [])


def load_task_splits(
    task_spec: dict[str, Any],
    *,
    seed: int,
    cache_dir: str | None,
) -> tuple[Dataset, Dataset, Dataset | None]:
    if task_spec["dataset_name"] == "ag_news":
        train_split = load_dataset("ag_news", split="train", cache_dir=cache_dir)
        test_split = load_dataset("ag_news", split="test", cache_dir=cache_dir)
        dev_ratio = task_spec.get("train_dev_split", 0.1)
        split = train_split.train_test_split(test_size=dev_ratio, seed=seed)
        return split["train"], split["test"], test_split

    dataset = load_dataset(
        task_spec["dataset_name"],
        task_spec.get("dataset_config"),
        cache_dir=cache_dir,
    )
    train_split = dataset[task_spec["train_split"]]
    dev_split = dataset[task_spec["dev_split"]]
    test_split = dataset[task_spec["test_split"]] if task_spec.get("test_split") else None
    return train_split, dev_split, test_split


def trim_split(split: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(split):
        return split
    return split.shuffle(seed=seed).select(range(max_samples))


def tokenize_classification_split(
    split: Dataset,
    *,
    tokenizer: GPT2Tokenizer,
    text_keys: list[str],
    label_key: str,
    max_seq_len: int,
) -> Dataset:
    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if len(text_keys) == 1:
            encoded = tokenizer(
                batch[text_keys[0]],
                padding="max_length",
                truncation=True,
                max_length=max_seq_len,
            )
        else:
            encoded = tokenizer(
                batch[text_keys[0]],
                batch[text_keys[1]],
                padding="max_length",
                truncation=True,
                max_length=max_seq_len,
            )
        encoded["labels"] = batch[label_key]
        return encoded

    tokenized = split.map(
        tokenize_batch,
        batched=True,
        remove_columns=split.column_names,
        desc="Tokenizing classification split",
    )
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


class OELMClassifier(nn.Module):
    def __init__(
        self,
        backbone: OELMForLanguageModeling,
        *,
        num_labels: int,
        setting_name: str,
        setting_spec: dict[str, Any],
        pooling: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.setting_name = setting_name
        self.probe_norm = nn.LayerNorm(backbone.config.d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone.config.d_model, num_labels)
        self._configure_trainability(setting_spec)

    def _configure_trainability(self, setting_spec: dict[str, Any]) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

        if setting_spec.get("unfreeze_layer_norms", False):
            for module in self.backbone.modules():
                if isinstance(module, nn.LayerNorm):
                    for param in module.parameters():
                        param.requires_grad = True

        top_blocks = int(setting_spec.get("unfreeze_top_blocks", 0))
        if top_blocks > 0:
            for block in self.backbone.blocks[-top_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        for param in self.probe_norm.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def trainable_param_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (hidden_states * mask).sum(dim=1) / denom

        last_indices = attention_mask.long().sum(dim=1).sub(1).clamp(min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_indices]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        hidden_states = self.backbone.forward_features(input_ids, attention_mask=attention_mask)
        pooled = self._pool(hidden_states, attention_mask)
        logits = self.classifier(self.dropout(self.probe_norm(pooled)))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"logits": logits, "loss": loss}


@torch.no_grad()
def evaluate_classifier(model: OELMClassifier, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    total_batches = 0
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += float(outputs["loss"].item())
        total_batches += 1
        preds = outputs["logits"].argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(total_batches, 1)
    metrics = {
        "loss": avg_loss,
        "accuracy": compute_accuracy(all_labels, all_preds),
        "weighted_f1": compute_weighted_f1(all_labels, all_preds),
    }
    if len(set(all_labels)) <= 2:
        metrics["binary_f1"] = compute_binary_f1(all_labels, all_preds)
    return metrics


def train_probe(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    task_spec, setting_spec, _ = load_task_spec(args.config_path, args.task, args.setting)
    run_name = build_run_name(args.run_name)
    output_root = (
        Path(args.output_root)
        / f"task={args.task}"
        / f"setting={args.setting}"
    )
    output_dir = build_run_dir(output_root, args.phase, args.method, args.seed, run_name)
    logger, log_path = setup_logging(output_dir, prefix=f"{args.phase}_{args.task}_{args.method}")

    start_time_utc = datetime.now(tz=timezone.utc)
    resource_usage = base_resource_usage(
        phase=args.phase,
        method=args.method,
        seed=args.seed,
        dataset=args.task,
        start_time_utc=start_time_utc,
        extra={"setting": args.setting, "output_dir": str(output_dir)},
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone, checkpoint_metadata = load_checkpoint(args.checkpoint, device=device)
    tokenizer = GPT2Tokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw, dev_raw, test_raw = load_task_splits(task_spec, seed=args.seed, cache_dir=args.cache_dir)
    train_raw = trim_split(train_raw, args.max_train_samples or task_spec.get("max_train_samples"), args.seed)
    dev_raw = trim_split(dev_raw, args.max_dev_samples or task_spec.get("max_dev_samples"), args.seed)
    if test_raw is not None:
        test_raw = trim_split(test_raw, args.max_test_samples, args.seed)

    train_dataset = tokenize_classification_split(
        train_raw,
        tokenizer=tokenizer,
        text_keys=task_spec["text_keys"],
        label_key=task_spec["label_key"],
        max_seq_len=args.max_seq_len,
    )
    dev_dataset = tokenize_classification_split(
        dev_raw,
        tokenizer=tokenizer,
        text_keys=task_spec["text_keys"],
        label_key=task_spec["label_key"],
        max_seq_len=args.max_seq_len,
    )
    test_dataset = None
    if test_raw is not None and args.eval_on_test:
        test_dataset = tokenize_classification_split(
            test_raw,
            tokenizer=tokenizer,
            text_keys=task_spec["text_keys"],
            label_key=task_spec["label_key"],
            max_seq_len=args.max_seq_len,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    model = OELMClassifier(
        backbone,
        num_labels=task_spec["num_labels"],
        setting_name=args.setting,
        setting_spec=setting_spec,
        pooling=args.pooling,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_training_steps = min(args.max_steps, args.num_epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, max(total_training_steps // 10, 1)),
        num_training_steps=max(total_training_steps, 1),
    )

    manifest = base_manifest(
        phase=args.phase,
        method=args.method,
        seed=args.seed,
        dataset=args.task,
        output_dir=output_dir,
        extra={
            "setting": args.setting,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config_path": str(Path(args.config_path).resolve()),
            "log_file": str(log_path),
        },
    )
    write_json(output_dir / "run_manifest.json", manifest)

    config_payload = {
        "phase": args.phase,
        "task": args.task,
        "setting": args.setting,
        "method": args.method,
        "seed": args.seed,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_metadata": checkpoint_metadata,
        "task_spec": task_spec,
        "setting_spec": setting_spec,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "patience": args.patience,
        "pooling": args.pooling,
        "dropout": args.dropout,
        "trainable_params": model.trainable_param_count(),
        "total_params": sum(param.numel() for param in model.parameters()),
        "train_samples": len(train_dataset),
        "dev_samples": len(dev_dataset),
        "test_samples": len(test_dataset) if test_dataset is not None else None,
    }
    write_json(output_dir / "config.json", config_payload)

    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_record: dict[str, Any] | None = None
    epochs_without_improvement = 0
    tokens_seen = 0
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        if global_step >= args.max_steps:
            break
        model.train()
        total_train_loss = 0.0
        total_batches = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for batch in progress:
            if global_step >= args.max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            batch_tokens = int(input_ids.numel())
            tokens_seen += batch_tokens
            total_train_loss += float(loss.item())
            total_batches += 1
            global_step += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}", "step": f"{global_step}/{args.max_steps}"})

        train_loss = total_train_loss / max(total_batches, 1)
        dev_metrics = evaluate_classifier(model, dev_loader, device)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
        }
        if "binary_f1" in dev_metrics:
            record["dev_binary_f1"] = dev_metrics["binary_f1"]
        history.append(record)
        logger.info(
            "Epoch %s | train_loss=%.4f | dev_accuracy=%.4f | dev_weighted_f1=%.4f",
            epoch,
            train_loss,
            dev_metrics["accuracy"],
            dev_metrics["weighted_f1"],
        )

        if dev_metrics["accuracy"] > best_accuracy:
            best_accuracy = dev_metrics["accuracy"]
            best_record = dict(record)
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "record": record,
                    "config": config_payload,
                },
                output_dir / "best_probe.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            logger.info("Early stopping triggered after %s epochs without improvement", args.patience)
            break

    summary = {
        "phase": args.phase,
        "task": args.task,
        "setting": args.setting,
        "method": args.method,
        "seed": args.seed,
        "best_epoch": best_record["epoch"] if best_record else None,
        "best_global_step": best_record["global_step"] if best_record else None,
        "best_accuracy": best_record["dev_accuracy"] if best_record else None,
        "best_weighted_f1": best_record["dev_weighted_f1"] if best_record else None,
        "history": history,
    }

    if test_loader is not None and best_record is not None:
        checkpoint = torch.load(output_dir / "best_probe.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate_classifier(model, test_loader, device)
        summary["test_metrics"] = test_metrics

    write_json(output_dir / "results.json", history)
    write_json(output_dir / "summary.json", summary)

    end_time_utc = datetime.now(tz=timezone.utc)
    finalize_resource_usage(
        resource_usage,
        end_time_utc=end_time_utc,
        state="completed",
        steps=global_step,
        tokens_seen=tokens_seen,
        peak_gpu_mem_gb=peak_gpu_memory_gb(),
        extra={
            "task": args.task,
            "setting": args.setting,
            "best_accuracy": summary["best_accuracy"],
            "best_weighted_f1": summary["best_weighted_f1"],
        },
    )
    write_json(output_dir / "resource_usage.json", resource_usage)
    manifest["completed_at_utc"] = end_time_utc.isoformat(timespec="seconds")
    manifest["state"] = "completed"
    write_json(output_dir / "run_manifest.json", manifest)


def main() -> None:
    args = parse_args()
    train_probe(args)


if __name__ == "__main__":
    main()
