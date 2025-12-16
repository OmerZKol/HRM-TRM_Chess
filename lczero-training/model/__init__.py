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
from model.hrm import HierarchicalReasoningModel_ACTV1
from model.bridge import AlphaZeroBridge

# Create and wrap model
hrm_model = HierarchicalReasoningModel_ACTV1(config_dict)
model = AlphaZeroBridge(hrm_model)

# Or use the convenient wrappers:
from model.ChessNNet import ChessNNet
model = ChessNNet(config)
```

Directory Structure:
-------------------
- common/: Shared utilities (layers, embeddings, initialization)
- hrm/: Hierarchical Reasoning Model implementations
- trm/: Tiny Recursive Model implementations
- heads/: Output head implementations (policy, value, moves left)
- bridge.py: AlphaZero training adapter
- ChessNNet.py, ChessTRMNet.py, ChessTRMBaselineNet.py: Training wrappers
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
from model.heads.value_heads import TensorFlowStyleValueHead, TensorFlowStyleMovesLeftHead

# Bridge
from model.bridge import AlphaZeroBridge

# Training wrappers (for backward compatibility)
from model.ChessNNet import ChessNNet
from model.ChessTRMNet import ChessTRMNet
from model.ChessTRMBaselineNet import ChessTRMBaselineNet

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
    "TensorFlowStyleValueHead",
    "TensorFlowStyleMovesLeftHead",
    # Bridge
    "AlphaZeroBridge",
    # Wrappers
    "ChessNNet",
    "ChessTRMNet",
    "ChessTRMBaselineNet",
]

__version__ = "2.0.0"
__author__ = "Chess AI Research Team"
