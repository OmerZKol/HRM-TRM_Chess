import sys
sys.path.append('..')
# sys.path.append('../../HRM_model')
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import torch
import torch.nn as nn
from model.HRM_model.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
sys.path.append('../..')
from model.HRMBridge import HRMAlphaZeroBridge

class ChessNNet(nn.Module):
    def __init__(self, config, board_size):
        super(ChessNNet, self).__init__()
        # game params
        self.board_size = board_size
        self.board_x, self.board_y = board_size
        self.action_size = config.get('action_size')
        hrm_config = config.get('hrm_config')
        # Initialize HRM model with bridge
        hrm_model = HierarchicalReasoningModel_ACTV1(hrm_config)
        self.bridge = HRMAlphaZeroBridge(hrm_model, self.action_size, self.board_size)

    def forward(self, s):
        # Use the bridge to handle HRM conversion
        pi, v, moves_left, q_info = self.bridge(s)

        # Return raw logits (loss function will apply log_softmax)
        # Don't apply log_softmax here - the loss function handles it
        return pi, torch.tanh(v), moves_left, q_info
