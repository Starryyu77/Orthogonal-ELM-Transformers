"""
OELM Fine-tuning from Pretrained Models - Phase 6 Multi-Dataset Version

支持数据集:
- imdb: 2分类情感分析
- ag_news: 4分类新闻
- sst2: 2分类短文本情感 (GLUE)
- xnli: 3分类自然语言推理
- mnli: 3分类自然语言推理 (GLUE)

使用方法:
    python finetune_multidata.py \
        --pretrained_path outputs/pretrain/oelm_qk_ffn/final_model.pt \
        --dataset ag_news \
        --output_dir outputs/phase6/ag_news/oelm_qk_ffn
"""

import os
import sys
import json
import time
import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent / "models"))


class OELMForSequenceClassification(nn.Module):
    """
    OELM for sequence classification (fine-tuning from pretrain)
    """

    def __init__(
        self,
        pretrained_model: nn.Module,
        num_classes: int,
        freeze_qk: bool = True,
        pool_method: str = "last",  # 'last' or 'mean'
    ):
        super().__init__()

        self.pretrained = pretrained_model
        self.num_classes = num_classes
        self.pool_method = pool_method

        # Remove LM head, add classification head
        d_model = pretrained_model.config.d_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes),
        )

        # Maintain freeze settings from pretrain
        if freeze_qk:
            print("Maintaining frozen Q/K from pretrain")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        # Get hidden states from pretrained model
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Get embeddings
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

            x = self.pretrained.token_embedding(input_ids)
            x = x + self.pretrained.position_embedding(positions)
            x = self.pretrained.dropout(x)

            # Pass through transformer blocks
            for block in self.pretrained.blocks:
                x = block(x)

            x = self.pretrained.ln_f(x)

        # Pool
        if self.pool_method == "last":
            pooled = x[:, -1, :]
        else:  # mean
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = x.mean(dim=1)

        # Classify
        logits = self.classifier(pooled)

        # Calculate loss
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"logits": logits, "loss": loss}


def load_pretrained_model(checkpoint_path: str, device: torch.device):
    """Load pretrained OELM model"""
    from modeling_oelm_pretrain import OELMForLanguageModeling, OELMConfig

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = OELMConfig(**checkpoint["config"])

    model = OELMForLanguageModeling(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded pretrained model from {checkpoint_path}")
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")

    return model


def load_dataset_for_classification(
    dataset_name: str,
    tokenizer: GPT2Tokenizer,
    max_length: int = 512,
    cache_dir: Optional[str] = None,
):
    """Load classification dataset - Phase 6 Multi-Dataset Version"""

    print(f"\nLoading dataset: {dataset_name}")

    if dataset_name == "imdb":
        dataset = load_dataset("imdb", cache_dir=cache_dir)
        num_classes = 2
        text_key = "text"
        label_key = "label"

    elif dataset_name in ["agnews", "ag_news"]:
        dataset = load_dataset("ag_news", cache_dir=cache_dir)
        num_classes = 4
        text_key = "text"
        label_key = "label"

    elif dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2", cache_dir=cache_dir)
        num_classes = 2
        text_key = "sentence"
        label_key = "label"

    elif dataset_name == "xnli":
        # XNLI英语版: premise + hypothesis
        dataset = load_dataset("xnli", "en", cache_dir=cache_dir)
        num_classes = 3
        # 使用validation作为测试集
        train_dataset = dataset["train"]
        test_dataset = dataset["validation"]

        def tokenize_function(examples):
            # Combine premise and hypothesis
            texts = [
                f"{p} </s> {h}"
                for p, h in zip(examples["premise"], examples["hypothesis"])
            ]
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        train_dataset = train_dataset.map(
            tokenize_function, batched=True, remove_columns=["premise", "hypothesis"]
        )
        test_dataset = test_dataset.map(
            tokenize_function, batched=True, remove_columns=["premise", "hypothesis"]
        )

        # Set format for PyTorch
        train_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )
        test_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )

        # Rename label column
        train_dataset = train_dataset.rename_column("label", "labels")
        test_dataset = test_dataset.rename_column("label", "labels")

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Num classes: {num_classes}")

        return train_dataset, test_dataset, num_classes

    elif dataset_name == "mnli":
        # MNLI: matched validation
        dataset = load_dataset("glue", "mnli", cache_dir=cache_dir)
        num_classes = 3
        train_dataset = dataset["train"]
        test_dataset = dataset["validation_matched"]

        def tokenize_function(examples):
            # Combine premise and hypothesis
            texts = [
                f"{p} </s> {h}"
                for p, h in zip(examples["premise"], examples["hypothesis"])
            ]
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        train_dataset = train_dataset.map(
            tokenize_function, batched=True, remove_columns=["premise", "hypothesis"]
        )
        test_dataset = test_dataset.map(
            tokenize_function, batched=True, remove_columns=["premise", "hypothesis"]
        )

        # Set format for PyTorch
        train_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )
        test_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )

        # Rename label column
        train_dataset = train_dataset.rename_column("label", "labels")
        test_dataset = test_dataset.rename_column("label", "labels")

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Num classes: {num_classes}")

        return train_dataset, test_dataset, num_classes

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Standard tokenization for non-NLI datasets
    def tokenize_function(examples):
        return tokenizer(
            examples[text_key],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    train_dataset = dataset["train"].map(
        tokenize_function, batched=True, remove_columns=[text_key]
    )

    test_dataset = dataset["test"].map(
        tokenize_function, batched=True, remove_columns=[text_key]
    )

    # Set format for PyTorch
    train_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", label_key]
    )
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", label_key])

    # Rename label column
    train_dataset = train_dataset.rename_column(label_key, "labels")
    test_dataset = test_dataset.rename_column(label_key, "labels")

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Num classes: {num_classes}")

    return train_dataset, test_dataset, num_classes


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += outputs["loss"].item()
            preds = torch.argmax(outputs["logits"], dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    model.train()
    return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}


def finetune(args):
    """Main fine-tuning function"""

    # Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load pretrained model
    print(f"\nLoading pretrained model from: {args.pretrained_path}")
    pretrained_model = load_pretrained_model(args.pretrained_path, device)

    # Load dataset
    train_dataset, test_dataset, num_classes = load_dataset_for_classification(
        args.dataset, tokenizer, args.max_seq_len, args.cache_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Create classification model
    model = OELMForSequenceClassification(
        pretrained_model,
        num_classes,
        freeze_qk=args.freeze_qk,
        pool_method=args.pool_method,
    )
    model = model.to(device)

    # Print model info
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nClassification model:")
    print(f"  Total params: {total:,}")
    print(f"  Trainable: {trainable:,} ({100 * trainable / total:.1f}%)")

    # Optimizer
    lr = args.lr if args.lr else (1e-3 if args.freeze_qk else 3e-4)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    print(f"\nTraining config:")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")

    # Training loop
    best_accuracy = 0
    results = []

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Evaluate
        eval_results = evaluate(model, test_loader, device)
        avg_loss = epoch_loss / len(train_loader)

        print(f"\nEpoch {epoch + 1}:")
        print(f"  Train loss: {avg_loss:.4f}")
        print(f"  Eval loss: {eval_results['loss']:.4f}")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  F1: {eval_results['f1']:.4f}")

        results.append({"epoch": epoch + 1, "train_loss": avg_loss, **eval_results})

        # Save best model
        if eval_results["accuracy"] > best_accuracy:
            best_accuracy = eval_results["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "accuracy": best_accuracy,
                    "epoch": epoch + 1,
                },
                output_dir / "best_model.pt",
            )
            print(f"  Saved best model (accuracy: {best_accuracy:.4f})")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFine-tuning completed!")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="OELM Fine-tuning - Phase 6 Multi-Dataset"
    )

    # Model
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--freeze_qk", action="store_true", help="Keep Q/K frozen from pretrain"
    )
    parser.add_argument(
        "--pool_method",
        type=str,
        default="last",
        choices=["last", "mean"],
        help="Pooling method",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="ag_news",
        choices=["imdb", "ag_news", "sst2", "xnli", "mnli"],
        help="Dataset for fine-tuning",
    )
    parser.add_argument("--max_seq_len", type=int, default=512)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # System
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./finetune_outputs")
    parser.add_argument("--cache_dir", type=str, default=None)

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("OELM Phase 6 Fine-tuning Configuration")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    finetune(args)


if __name__ == "__main__":
    main()
