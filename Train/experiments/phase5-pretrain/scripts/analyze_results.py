#!/usr/bin/env python3
"""
OELM Pretrain Experiment Analysis Script

分析预训练和下游微调结果
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PretrainResult:
    """预训练结果"""

    method: str
    final_loss: float
    final_ppl: float
    best_loss: float
    best_ppl: float
    total_steps: int
    trainable_params: int
    frozen_params: int
    total_params: int

    @property
    def trainable_ratio(self) -> float:
        return 100 * self.trainable_params / self.total_params


@dataclass
class FinetuneResult:
    """微调结果"""

    method: str
    dataset: str
    best_accuracy: float
    best_f1: float
    final_accuracy: float
    final_f1: float
    epochs: int


class ExperimentAnalyzer:
    """实验分析器"""

    def __init__(self, base_dir: str = "/projects/LlamaFactory/OELM-Pretrain"):
        self.base_dir = Path(base_dir)
        self.pretrain_results: Dict[str, PretrainResult] = {}
        self.finetune_results: Dict[str, List[FinetuneResult]] = {}

    def load_pretrain_result(self, method: str) -> Optional[PretrainResult]:
        """加载预训练结果"""
        output_dir = self.base_dir / "outputs" / "pretrain" / method

        # Find checkpoint
        best_path = output_dir / "best" / "pytorch_model.pt"
        if not best_path.exists():
            return None

        # Load checkpoint to get info
        try:
            import torch

            checkpoint = torch.load(best_path, map_location="cpu")

            config = checkpoint.get("config", {})

            # Calculate params
            total_params = (
                config.get("vocab_size", 50257) * config.get("d_model", 768) * 2
            )  # embeddings
            d_model = config.get("d_model", 768)
            num_layers = config.get("num_layers", 12)
            num_heads = config.get("num_heads", 12)
            d_ff = config.get("d_ff", 3072)

            # Per layer
            attn_params = 4 * d_model * d_model
            ffn_params = 2 * d_model * d_ff
            ln_params = 4 * d_model
            layer_params = attn_params + ffn_params + ln_params
            total_params += num_layers * layer_params

            # Estimate frozen params
            freeze_qk = config.get("freeze_qk", False)
            freeze_ffn = config.get("freeze_ffn", False)

            frozen_params = 0
            if freeze_qk:
                frozen_params += 2 * d_model * d_model  # Q + K
            if freeze_ffn:
                frozen_params += 2 * d_model * d_ff  # FFN up + down
            frozen_params *= num_layers

            trainable_params = total_params - frozen_params

            return PretrainResult(
                method=method,
                final_loss=checkpoint.get("loss", 0),
                final_ppl=self._compute_ppl(checkpoint.get("loss", 0)),
                best_loss=checkpoint.get("loss", 0),
                best_ppl=self._compute_ppl(checkpoint.get("loss", 0)),
                total_steps=checkpoint.get("step", 0),
                trainable_params=trainable_params,
                frozen_params=frozen_params,
                total_params=total_params,
            )
        except Exception as e:
            print(f"Error loading {method}: {e}")
            return None

    def load_finetune_result(
        self, method: str, dataset: str
    ) -> Optional[FinetuneResult]:
        """加载微调结果"""
        result_path = (
            self.base_dir
            / "outputs"
            / "finetune"
            / f"{method}_{dataset}"
            / "results.json"
        )

        if not result_path.exists():
            return None

        try:
            with open(result_path) as f:
                data = json.load(f)

            # Find best epoch
            best_epoch = max(data, key=lambda x: x["accuracy"])
            final_epoch = data[-1]

            return FinetuneResult(
                method=method,
                dataset=dataset,
                best_accuracy=best_epoch["accuracy"],
                best_f1=best_epoch["f1"],
                final_accuracy=final_epoch["accuracy"],
                final_f1=final_epoch["f1"],
                epochs=len(data),
            )
        except Exception as e:
            print(f"Error loading finetune result: {e}")
            return None

    def analyze_all(self):
        """分析所有结果"""
        print("=" * 70)
        print("OELM Pretrain Experiment Analysis")
        print("=" * 70)

        # Load pretrain results
        print("\n## Pre-training Results\n")

        for method in ["baseline", "oelm_qk", "oelm_qk_ffn"]:
            result = self.load_pretrain_result(method)
            if result:
                self.pretrain_results[method] = result
                self._print_pretrain_result(result)

        # Compare pretrain results
        if len(self.pretrain_results) > 1:
            self._compare_pretrain()

        # Load finetune results
        print("\n## Downstream Fine-tuning Results\n")

        for method in ["baseline", "oelm_qk", "oelm_qk_ffn"]:
            result = self.load_finetune_result(method, "imdb")
            if result:
                if method not in self.finetune_results:
                    self.finetune_results[method] = []
                self.finetune_results[method].append(result)
                self._print_finetune_result(result)

        # Compare finetune results
        if len(self.finetune_results) > 1:
            self._compare_finetune()

    def _print_pretrain_result(self, result: PretrainResult):
        """打印预训练结果"""
        print(f"### {result.method.upper()}")
        print(f"  Final Loss: {result.final_loss:.4f}")
        print(f"  Final PPL:  {result.final_ppl:.2f}")
        print(f"  Steps:      {result.total_steps}")
        print(f"  Params:     {result.total_params:,} total")
        print(
            f"              {result.trainable_params:,} trainable ({result.trainable_ratio:.1f}%)"
        )
        print(
            f"              {result.frozen_params:,} frozen ({100 - result.trainable_ratio:.1f}%)"
        )
        print()

    def _print_finetune_result(self, result: FinetuneResult):
        """打印微调结果"""
        print(f"### {result.method.upper()} - {result.dataset.upper()}")
        print(
            f"  Best Accuracy: {result.best_accuracy:.4f} ({result.best_accuracy * 100:.2f}%)"
        )
        print(f"  Best F1:       {result.best_f1:.4f}")
        print(
            f"  Final Accuracy: {result.final_accuracy:.4f} ({result.final_accuracy * 100:.2f}%)"
        )
        print(f"  Epochs:        {result.epochs}")
        print()

    def _compare_pretrain(self):
        """对比预训练结果"""
        print("## Pre-training Comparison\n")

        baseline = self.pretrain_results.get("baseline")
        if not baseline:
            print("Baseline not found, cannot compare")
            return

        for method, result in self.pretrain_results.items():
            if method == "baseline":
                continue

            ppl_diff = (
                (result.final_ppl - baseline.final_ppl) / baseline.final_ppl
            ) * 100
            param_diff = baseline.trainable_params - result.trainable_params
            param_ratio = 100 * result.trainable_params / baseline.trainable_params

            print(f"### {method.upper()} vs Baseline")
            print(
                f"  PPL:           {result.final_ppl:.2f} vs {baseline.final_ppl:.2f} ({ppl_diff:+.1f}%)"
            )
            print(
                f"  Params:        {result.trainable_ratio:.1f}% vs {baseline.trainable_ratio:.1f}%"
            )
            print(f"  Param Savings: {param_diff:,} ({100 - param_ratio:.1f}%)")

            if abs(ppl_diff) < 10:
                print(f"  Status:        ✅ Within 10% PPL target")
            else:
                print(f"  Status:        ⚠️  Exceeds 10% PPL threshold")
            print()

    def _compare_finetune(self):
        """对比微调结果"""
        print("## Fine-tuning Comparison\n")

        baseline = None
        for method, results in self.finetune_results.items():
            if method == "baseline" and results:
                baseline = results[0]
                break

        if not baseline:
            print("Baseline not found, cannot compare")
            return

        for method, results in self.finetune_results.items():
            if method == "baseline":
                continue

            for result in results:
                acc_diff = (result.best_accuracy - baseline.best_accuracy) * 100

                print(f"### {method.upper()} vs Baseline")
                print(
                    f"  Accuracy: {result.best_accuracy * 100:.2f}% vs {baseline.best_accuracy * 100:.2f}% ({acc_diff:+.2f}%)"
                )

                if acc_diff >= -2:
                    print(f"  Status:   ✅ Within -2% accuracy target")
                else:
                    print(f"  Status:   ❌ Below -2% accuracy threshold")
                print()

    @staticmethod
    def _compute_ppl(loss: float) -> float:
        """Compute perplexity from loss"""
        import math

        return math.exp(loss) if loss < 10 else float("inf")

    def generate_report(self, output_file: str = "experiment_report.md"):
        """生成实验报告"""
        # This would save the analysis to a markdown file
        pass


def main():
    """Main entry point"""
    analyzer = ExperimentAnalyzer()
    analyzer.analyze_all()

    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
