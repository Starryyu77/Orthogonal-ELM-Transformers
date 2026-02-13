"""
Phase 4: GPT Classification Models

包含:
- GPTForSequenceClassification: Baseline分类模型
- OELMForSequenceClassification: OELM分类模型 (冻结Q/K)
"""

from .modeling_gpt_classification import (
    GPTForSequenceClassification,
    create_gpt_classifier
)

from .modeling_oelm_classification import (
    OELMForSequenceClassification,
    create_oelm_classifier
)

__all__ = [
    'GPTForSequenceClassification',
    'create_gpt_classifier',
    'OELMForSequenceClassification',
    'create_oelm_classifier',
]