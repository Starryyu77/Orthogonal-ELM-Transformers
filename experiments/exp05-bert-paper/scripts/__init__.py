"""
BERT Reservoir Test Package

This package implements the Head-wise Orthogonal initialization for BERT
to verify the Reservoir Computing hypothesis.
"""

from .modeling_bert_oelm import (
    load_bert_with_head_wise_orthogonal,
    apply_head_wise_orthogonal_,
    check_orthogonality,
    freeze_model_parameters,
    print_trainable_parameters,
)

__all__ = [
    'load_bert_with_head_wise_orthogonal',
    'apply_head_wise_orthogonal_',
    'check_orthogonality',
    'freeze_model_parameters',
    'print_trainable_parameters',
]
