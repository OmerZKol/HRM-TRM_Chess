"""
TRM Baseline (Single-level transformer) wrapper for chess training.

This wrapper provides backward compatibility for training scripts.
"""

import torch.nn as nn

from model.trm import TinyRecursiveReasoningModel_ACTV1
from model.bridge import AlphaZeroBridge


class ChessTRMBaselineNet(nn.Module):
    """
    TRM Baseline wrapper for chess training with AlphaZero-style outputs.

    The baseline is a simplified version without hierarchical H/L split or inner cycles.

    Value output is raw WDL logits - softmax is applied in the loss function.
    """

    def __init__(self, config):
        super(ChessTRMBaselineNet, self).__init__()
        trm_config = config.get('trm_config')

        # Initialize TRM baseline model with bridge
        trm_baseline_model = TinyRecursiveReasoningModel_ACTV1(trm_config)
        self.bridge = AlphaZeroBridge(trm_baseline_model)

    def forward(self, s):
        """
        Forward pass through TRM baseline model.

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
