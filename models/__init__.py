"""
Orthogonal ELM Transformer Models

This package contains implementations of:
- OrthogonalELMTransformer: Transformer with frozen orthogonal Q/K projections
- GPT: Standard GPT model for comparison
"""

from .modeling_oelm import (
    OrthogonalLinear,
    OrthogonalMultiHeadAttention,
    OrthogonalTransformerLayer,
    OrthogonalELMTransformer,
    create_oelm_tiny,
    create_oelm_small,
    create_oelm_medium,
    create_oelm_large,
)

from .modeling_gpt import (
    MultiHeadAttention,
    GPTTransformerLayer,
    GPT,
    create_gpt_tiny,
    create_gpt2_small,
    create_gpt2_medium,
    create_gpt2_large,
    create_gpt2_xl,
)

__all__ = [
    # OELM models
    'OrthogonalLinear',
    'OrthogonalMultiHeadAttention',
    'OrthogonalTransformerLayer',
    'OrthogonalELMTransformer',
    'create_oelm_tiny',
    'create_oelm_small',
    'create_oelm_medium',
    'create_oelm_large',
    # GPT models
    'MultiHeadAttention',
    'GPTTransformerLayer',
    'GPT',
    'create_gpt_tiny',
    'create_gpt2_small',
    'create_gpt2_medium',
    'create_gpt2_large',
    'create_gpt2_xl',
]
