import sys
sys.path.append('..')
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import torch
import torch.nn as nn

# Add TinyRecursiveModels to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'TinyRecursiveModels')))

from model.trm_model.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from model.HRMBridge import HRMAlphaZeroBridge


class ChessTRMNet(nn.Module):
    """
    TRM (Tiny Recursive Model) wrapper for chess training.
    Uses the adapted TinyRecursiveReasoningModel_ACTV1 with chess tokenization.
    """
    def __init__(self, config, board_size):
        super(ChessTRMNet, self).__init__()
        # Game params
        self.board_size = board_size
        self.board_x, self.board_y = board_size
        self.action_size = config.get('action_size')
        trm_config = config.get('trm_config')

        # Initialize TRM model with bridge
        trm_model = TinyRecursiveReasoningModel_ACTV1(trm_config)
        self.bridge = HRMAlphaZeroBridge(trm_model, self.action_size, self.board_size)

    def forward(self, s):
        # Use the bridge to handle TRM conversion
        pi, v, moves_left, q_info = self.bridge(s)

        # Return in AlphaZero format: log probabilities and tanh values
        return pi, torch.tanh(v), moves_left, q_info
