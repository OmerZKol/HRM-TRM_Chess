"""
Hierarchical Reasoning Model (HRM) wrapper for chess training.

This wrapper provides backward compatibility for training scripts.
"""

import torch.nn as nn

from model.hrm.hrm_model import HierarchicalReasoningModel_ACTV1
from model.bridge import AlphaZeroBridge


class ChessNNet(nn.Module):
    """
    HRM wrapper for chess training with AlphaZero-style outputs.

    Value output is raw WDL logits - softmax is applied in the loss function.
    """

    def __init__(self, config):
        super(ChessNNet, self).__init__()
        hrm_config = config.get('hrm_config')
        # Initialize HRM model with bridge
        hrm_model = HierarchicalReasoningModel_ACTV1(hrm_config)
        self.bridge = AlphaZeroBridge(hrm_model)

    def forward(self, s):
        """
        Forward pass through HRM model.

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
