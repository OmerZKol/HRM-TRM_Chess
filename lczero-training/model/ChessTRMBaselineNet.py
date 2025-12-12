import sys
sys.path.append('..')
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import torch
import torch.nn as nn

# Add TinyRecursiveModels to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'TinyRecursiveModels')))

from model.trm_model.recursive_reasoning.transformers_baseline import TinyRecursiveReasoningModel_ACTV1
from model.HRMBridge2 import HRMAlphaZeroBridge2


class ChessTRMBaselineNet(nn.Module):
    """
    TRM Baseline (Single-level transformer) wrapper for chess training.
    Uses the baseline transformer from transformers_baseline.py - a simplified
    version without hierarchical H/L split or inner cycles.
    """
    def __init__(self, config, board_size):
        super(ChessTRMBaselineNet, self).__init__()
        # Game params
        self.board_size = board_size
        self.board_x, self.board_y = board_size
        self.action_size = config.get('action_size')
        trm_config = config.get('trm_config')

        # Initialize TRM baseline model with bridge
        trm_baseline_model = TinyRecursiveReasoningModel_ACTV1(trm_config)
        self.bridge = HRMAlphaZeroBridge2(trm_baseline_model, self.action_size, self.board_size)

    def forward(self, s):
        # Use the bridge to handle TRM conversion
        pi, v, moves_left, q_info = self.bridge(s)

        # Return in AlphaZero format: log probabilities and tanh values
        return pi, torch.tanh(v), moves_left, q_info
