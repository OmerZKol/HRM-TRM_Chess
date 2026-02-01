"""
Output head implementations for chess models.
"""

from model.heads.attention_policy import AttentionPolicyHead
from model.heads.value_heads import TensorFlowStyleValueHead, TensorFlowStyleMovesLeftHead

__all__ = [
    "AttentionPolicyHead",
    "TensorFlowStyleValueHead",
    "TensorFlowStyleMovesLeftHead",
]
