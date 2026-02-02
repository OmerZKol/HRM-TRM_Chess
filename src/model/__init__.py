"""
Chess Model Library

This package contains implementations of Hierarchical Reasoning Models (HRM) and
Tiny Recursive Models (TRM) for chess, adapted to work with AlphaZero-style training.

Main Models:
-----------
- HierarchicalReasoningModel_ACTV1: Full HRM with hierarchical H/L levels
- TinyRecursiveModel_ACTV1: TRM with recursive reasoning
- TinyRecursiveReasoningModel_ACTV1: TRM baseline (single-level transformer)

Usage Example:
-------------
```python
from model.ChessModelWrapper import ChessModelWrapper
model = ChessModelWrapper(config)  # config['model_type'] determines which model
```

Directory Structure:
-------------------
- common/: Shared utilities (layers, embeddings, initialization)
- hrm/: Hierarchical Reasoning Model implementations
- trm/: Tiny Recursive Model implementations
- heads/: Output head implementations (policy, value, moves left)
- bridge.py: training adapter
- ChessModelWrapper.py: Unified training wrapper
"""

# Common utilities
from model.common.initialization import trunc_normal_init_
from model.common.layers import (
    CastedLinear,
    CastedEmbedding,
    RotaryEmbedding,
    Attention,
    LinearSwish,
    SwiGLU,
    rms_norm,
)
from model.common.sparse_embedding import CastedSparseEmbedding

# HRM models
from model.hrm.hrm_model import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
)

# TRM models
from model.trm import (
    TinyRecursiveModel_ACTV1,
    TinyRecursiveModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)

# Output heads
from model.heads.attention_policy import AttentionPolicyHead
from model.heads.value_heads import ValueHead, MovesLeftHead

# Bridge
from model.bridge import AlphaZeroBridge

# Unified wrapper
from model.ChessModelWrapper import ChessModelWrapper

__all__ = [
    # Common
    "trunc_normal_init_",
    "CastedLinear",
    "CastedEmbedding",
    "RotaryEmbedding",
    "Attention",
    "LinearSwish",
    "SwiGLU",
    "rms_norm",
    "CastedSparseEmbedding",
    # HRM
    "HierarchicalReasoningModel_ACTV1",
    "HierarchicalReasoningModel_ACTV1Config",
    # TRM
    "TinyRecursiveModel_ACTV1",
    "TinyRecursiveModel_ACTV1Config",
    "TinyRecursiveReasoningModel_ACTV1",
    "TinyRecursiveReasoningModel_ACTV1Config",
    # Heads
    "AttentionPolicyHead",
    "ValueHead",
    "MovesLeftHead",
    # Bridge
    "AlphaZeroBridge",
    # Wrapper
    "ChessModelWrapper",
]