"""
Output head implementations for chess models.
"""

from model.heads.attention_policy import AttentionPolicyHead
from model.heads.value_heads import ValueHead, MovesLeftHead, LinearPolicyHead, CombinedHeads

__all__ = [
    "AttentionPolicyHead",
    "ValueHead",
    "MovesLeftHead",
    "LinearPolicyHead",
    "CombinedHeads",
]
