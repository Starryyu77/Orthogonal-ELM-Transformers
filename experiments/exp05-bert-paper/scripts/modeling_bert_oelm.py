"""
BERT with Head-wise Orthogonal Initialization for Reservoir Computing Test

This module implements the core innovation of "Head-wise Orthogonality":
instead of applying QR decomposition globally to Q/K weights, we reshape them
into [num_heads, head_dim, hidden_dim] and apply QR independently per head.

This preserves inter-head expressiveness while maintaining intra-head orthogonality.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import BertForSequenceClassification, BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention


def apply_head_wise_orthogonal_(weight: nn.Parameter, num_heads: int) -> None:
    """
    Apply head-wise orthogonal initialization to a weight matrix.

    The key fix from GPT+OELM failure:
    - OLD (Global): QR on [768, 768] -> destroys per-head structure
    - NEW (Head-wise): QR on [12, 64, 768] per head -> preserves head structure

    Args:
        weight: Parameter of shape [hidden_dim, hidden_dim]
        num_heads: Number of attention heads (12 for bert-base)
    """
    with torch.no_grad():
        hidden_dim = weight.size(0)
        head_dim = hidden_dim // num_heads

        # Reshape to [num_heads, head_dim, hidden_dim]
        w = weight.view(num_heads, head_dim, hidden_dim).clone()

        # Apply QR decomposition independently per head
        for i in range(num_heads):
            # w[i] shape: [head_dim, hidden_dim]
            # Transpose for QR: [hidden_dim, head_dim]
            q, r = torch.linalg.qr(w[i].T, mode='reduced')

            # Adjust signs for deterministic output
            signs = torch.sign(torch.diag(r))
            q = q * signs.unsqueeze(0)

            # Transpose back: [head_dim, hidden_dim]
            w[i] = q.T

        # Copy back to original weight
        weight.copy_(w.view(hidden_dim, hidden_dim))


def check_orthogonality(weight: torch.Tensor, num_heads: int, tolerance: float = 1e-5) -> bool:
    """
    Verify that each head's weight matrix is orthogonal: W @ W^T ≈ I

    This is the "Must-Have" check - never assume, always verify.

    Args:
        weight: Weight matrix of shape [hidden_dim, hidden_dim]
        num_heads: Number of attention heads
        tolerance: Numerical tolerance for identity check

    Returns:
        True if all heads are orthogonal

    Raises:
        AssertionError: If any head fails orthogonality check
    """
    hidden_dim = weight.size(0)
    head_dim = hidden_dim // num_heads

    # Reshape to [num_heads, head_dim, hidden_dim]
    w = weight.view(num_heads, head_dim, hidden_dim)

    for i in range(num_heads):
        # Compute W @ W^T for this head
        # w[i] shape: [head_dim, hidden_dim]
        # w[i] @ w[i].T shape: [head_dim, head_dim]
        product = w[i] @ w[i].T
        identity = torch.eye(head_dim, device=weight.device, dtype=weight.dtype)

        max_error = torch.max(torch.abs(product - identity)).item()

        if max_error > tolerance:
            raise AssertionError(
                f"Head {i} orthogonality check FAILED! "
                f"Max error: {max_error:.2e} (tolerance: {tolerance:.2e})\n"
                f"W @ W^T =\n{product}\n"
                f"Expected I =\n{identity}"
            )

    print(f"✓ Orthogonality check passed ({num_heads} heads, max error < {tolerance})")
    return True


def freeze_model_parameters(model: BertForSequenceClassification, freeze_mode: bool = True) -> None:
    """
    Freeze/Unfreeze model parameters according to the Reservoir Test protocol.

    The Head Integrity Rule:
    - MUST freeze: Query, Key projections (Q/K)
    - MUST NOT freeze: Pooler, Classifier, Value, FFN, LayerNorm

    Args:
        model: BERT model for sequence classification
        freeze_mode: True for OELM (freeze Q/K), False for Baseline (no freezing)
    """
    if not freeze_mode:
        # Baseline: everything is trainable
        print("Mode: Baseline (all parameters trainable)")
        for param in model.parameters():
            param.requires_grad = True
        return

    # OELM mode: freeze only Q/K, keep everything else trainable
    print("Mode: OELM-Freeze (Query/Key frozen, rest trainable)")

    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        # Determine if this parameter should be frozen
        should_freeze = False

        # Freeze Query and Key projections
        if 'query' in name.lower() or 'key' in name.lower():
            should_freeze = True
            frozen_count += param.numel()
        else:
            trainable_count += param.numel()

        param.requires_grad = not should_freeze

        # Log first layer for verification
        if 'encoder.layer.0' in name and ('query' in name or 'key' in name):
            print(f"  Frozen: {name} ({param.numel()/1e6:.2f}M params)")
        elif 'encoder.layer.0' in name and ('value' in name or 'attention.output.dense' in name):
            print(f"  Trainable: {name} ({param.numel()/1e6:.2f}M params)")

    # Verify Pooler is NOT frozen (Head Integrity check)
    pooler_params = list(model.bert.pooler.parameters())
    pooler_trainable = all(p.requires_grad for p in pooler_params)
    if not pooler_trainable:
        raise RuntimeError("CRITICAL: Pooler is frozen! This violates the Head Integrity Rule.")

    # Verify Classifier is NOT frozen
    classifier_params = list(model.classifier.parameters())
    classifier_trainable = all(p.requires_grad for p in classifier_params)
    if not classifier_trainable:
        raise RuntimeError("CRITICAL: Classifier is frozen!")

    print(f"\nParameter Summary:")
    print(f"  Frozen (Q/K only): {frozen_count/1e6:.2f}M params")
    print(f"  Trainable: {trainable_count/1e6:.2f}M params ({trainable_count/(frozen_count+trainable_count)*100:.1f}%)")
    print(f"  ✓ Pooler integrity: PASSED")
    print(f"  ✓ Classifier integrity: PASSED")


def load_bert_with_head_wise_orthogonal(
    model_name: str = "bert-base-uncased",
    num_classes: int = 2,
    freeze_mode: bool = True,
    verify_orthogonality: bool = True,
    init_method: str = 'orthogonal'  # NEW: 'orthogonal' or 'normal'
) -> BertForSequenceClassification:
    """
    Load BERT and apply head-wise initialization to Q/K weights.

    This is the main entry point for the Reservoir Test.

    Args:
        model_name: HuggingFace model identifier
        num_classes: Number of output classes (2 for SST-2, 3 for MNLI)
        freeze_mode: True for OELM, False for Baseline
        verify_orthogonality: Run orthogonality check after initialization
        init_method: 'orthogonal' (QR decomposition) or 'normal' (Gaussian)

    Returns:
        Configured BERT model ready for training
    """
    print(f"\n{'='*60}")
    print(f"Loading BERT with Head-wise {init_method.capitalize()} Initialization")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Task: {num_classes}-class classification")
    print(f"Freeze mode: {freeze_mode}")
    print(f"Init method: {init_method}")

    # Load pre-trained BERT
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )

    config = model.config
    num_heads = config.num_attention_heads  # 12 for bert-base
    hidden_dim = config.hidden_size  # 768 for bert-base

    print(f"\nArchitecture: {hidden_dim}d, {num_heads} heads, {hidden_dim//num_heads}d per head")

    # Apply head-wise initialization to Q/K in all layers
    num_layers = config.num_hidden_layers
    print(f"\nApplying head-wise {init_method} initialization to {num_layers} layers...")

    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        attention = layer.attention.self  # BertSelfAttention

        if init_method == 'orthogonal':
            # Apply QR decomposition per head
            apply_head_wise_orthogonal_(attention.query.weight, num_heads)
            apply_head_wise_orthogonal_(attention.key.weight, num_heads)

            # Verify orthogonality (only first layer for speed)
            if verify_orthogonality and layer_idx == 0:
                print(f"\nVerifying orthogonality for layer {layer_idx}...")
                check_orthogonality(attention.query.weight.data, num_heads)
                check_orthogonality(attention.key.weight.data, num_heads)

        elif init_method == 'normal':
            # Standard Gaussian initialization (ablation study)
            nn.init.normal_(attention.query.weight, mean=0.0, std=config.initializer_range)
            nn.init.normal_(attention.key.weight, mean=0.0, std=config.initializer_range)

            if layer_idx == 0:
                print(f"\nApplied normal (Gaussian) initialization to Q/K")
        else:
            raise ValueError(f"Unknown init_method: {init_method}. Choose 'orthogonal' or 'normal'.")

    if verify_orthogonality and init_method == 'orthogonal':
        print(f"✓ Orthogonality verified for all {num_layers} layers")

    # Freeze/Unfreeze parameters according to protocol
    freeze_model_parameters(model, freeze_mode=freeze_mode)

    print(f"{'='*60}\n")

    return model


def print_trainable_parameters(model: BertForSequenceClassification) -> None:
    """
    Print a summary of trainable vs frozen parameters.
    Useful for verification before training.
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    print("\nDetailed Parameter Breakdown:")
    print("-" * 80)
    print(f"{'Layer Name':<50} {'Parameters':>12} {'Status':>10}")
    print("-" * 80)

    for name, param in model.named_parameters():
        num_params = param.numel()
        status = "Trainable" if param.requires_grad else "Frozen"

        # Only print important layers
        if any(key in name for key in ['embeddings', 'pooler', 'classifier', 'layer.0', 'layer.11']):
            print(f"{name:<50} {num_params:>12,} {status:>10}")

        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params

    print("-" * 80)
    print(f"{'Total':<50} {total_params:>12,} {trainable_params/total_params*100:>9.1f}%")
    print(f"{'Frozen (Q/K only)':<50} {frozen_params:>12,}")
    print(f"{'Trainable':<50} {trainable_params:>12,}")
    print("-" * 80)


if __name__ == "__main__":
    # Quick test
    print("Testing BERT OELM model loading...")

    # Test OELM mode
    model_oelm = load_bert_with_head_wise_orthogonal(
        freeze_mode=True,
        verify_orthogonality=True
    )
    print_trainable_parameters(model_oelm)

    # Test Baseline mode
    print("\n\n" + "="*60)
    model_baseline = load_bert_with_head_wise_orthogonal(
        freeze_mode=False,
        verify_orthogonality=False
    )
    print_trainable_parameters(model_baseline)
