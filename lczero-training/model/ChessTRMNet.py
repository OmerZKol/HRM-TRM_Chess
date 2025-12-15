"""
Tiny Recursive Model (TRM) wrapper for chess training.

This wrapper provides backward compatibility for training scripts.
It applies torch.tanh() to the value output as expected by the training code.
"""

import torch
import torch.nn as nn

from model.trm.trm_model import TinyRecursiveModel_ACTV1
from model.bridge import AlphaZeroBridge


class ChessTRMNet(nn.Module):
    """
    TRM wrapper for chess training with AlphaZero-style outputs.

    Note: The torch.tanh() applied to value outputs is intentional for compatibility
    with the training pipeline.
    """

    def __init__(self, config):
        super(ChessTRMNet, self).__init__()
        trm_config = config.get('trm_config')

        # Initialize TRM model with bridge
        trm_model = TinyRecursiveModel_ACTV1(trm_config)
        self.bridge = AlphaZeroBridge(trm_model)

    def forward(self, s):
        """
        Forward pass through TRM model.

        Args:
            s: Board state [batch_size, 112, 8, 8]

        Returns:
            pi: Policy logits [batch_size, action_size]
            v: Value predictions with tanh applied [batch_size, 3]
            moves_left: Moves left predictions [batch_size, 1]
            q_info: Dictionary with Q-learning information
        """
        pi, v, moves_left, q_info = self.bridge(s)

        # Apply tanh to value output for training compatibility
        return pi, torch.tanh(v), moves_left, q_info
