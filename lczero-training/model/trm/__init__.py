"""
Tiny Recursive Model (TRM) implementation.
"""

from model.trm.trm_model import (
    TinyRecursiveReasoningModel_ACTV1 as TinyRecursiveModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config as TinyRecursiveModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1Carry as TinyRecursiveModel_ACTV1Carry,
)
from model.trm.trm_baseline import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1Carry,
)

__all__ = [
    "TinyRecursiveModel_ACTV1",
    "TinyRecursiveModel_ACTV1Config",
    "TinyRecursiveModel_ACTV1Carry",
    "TinyRecursiveReasoningModel_ACTV1",
    "TinyRecursiveReasoningModel_ACTV1Config",
    "TinyRecursiveReasoningModel_ACTV1Carry",
    "TinyRecursiveModel_ACTV1_Adapter",
    "TinyRecursiveModel_ACTV1_AdapterConfig",
    "TinyRecursiveModel_ACTV1_AdapterCarry",
]
