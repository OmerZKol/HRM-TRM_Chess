"""
TRM with Adapter (concat adapter for recursion) wrapper for chess training.

This wrapper provides backward compatibility for training scripts.
"""

import torch.nn as nn

from model.trm import TinyRecursiveModel_ACTV1_Adapter
from model.bridge import AlphaZeroBridge


class ChessTRMAdapterNet(nn.Module):
    """
    TRM with Adapter wrapper for chess training with AlphaZero-style outputs.

    This model uses a concat adapter to combine recursion output with input
    instead of simple addition.

    Value output is raw WDL logits - softmax is applied in the loss function.
    """

    def __init__(self, config):
        super(ChessTRMAdapterNet, self).__init__()
        trm_config = config.get('trm_config')

        # Initialize TRM adapter model with bridge
        trm_adapter_model = TinyRecursiveModel_ACTV1_Adapter(trm_config)
        self.bridge = AlphaZeroBridge(trm_adapter_model)

    def forward(self, s):
        """
        Forward pass through TRM adapter model.

        Args:
            s: Board state [batch_size, 112, 8, 8]

        Returns:
            pi: Policy logits [batch_size, action_size]
            v: Value logits in WDL format [batch_size, 3] (softmax applied in loss)
            moves_left: Moves left predictions [batch_size, 1]
            q_info: Dictionary with Q-learning information
        """
        pi, v, moves_left, q_info = self.bridge(s)

        return pi, v, moves_left, q_info
