"""
Common utilities shared across HRM and TRM models.
"""

from model.common.initialization import trunc_normal_init_
from model.common.layers import (
    CastedLinear,
    CastedEmbedding,
    RotaryEmbedding,
    Attention,
    LinearSwish,
    SwiGLU,
    rms_norm,
    apply_rotary_pos_emb,
    rotate_half,
)
from model.common.sparse_embedding import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)

__all__ = [
    "trunc_normal_init_",
    "CastedLinear",
    "CastedEmbedding",
    "RotaryEmbedding",
    "Attention",
    "LinearSwish",
    "SwiGLU",
    "rms_norm",
    "apply_rotary_pos_emb",
    "rotate_half",
    "CastedSparseEmbedding",
    "CastedSparseEmbeddingSignSGD_Distributed",
]
