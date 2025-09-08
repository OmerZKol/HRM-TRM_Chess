import sys
sys.path.append('..')
# sys.path.append('../../HRM_model')
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.HRM_model.models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
sys.path.append('../..')
from model.HRMBridge import HRMAlphaZeroBridge

class ChessNNet(nn.Module):
    def __init__(self, board_size, action_size, batch_size=32):
        super(ChessNNet, self).__init__()
        # game params
        self.board_size = board_size
        self.board_x, self.board_y = board_size
        self.action_size = action_size
        self.batch_size = batch_size

        # Optimized HRM model configuration for Chess
        hrm_config = {
            'batch_size': self.batch_size,
            'seq_len': self.board_x * self.board_y,  # Treat each board square as a token
            'puzzle_emb_ndim': 0,  # No puzzle embeddings for Chess
            'num_puzzle_identifiers': 1,
            'vocab_size': 3, # placeholder tbh
            
            # Optimized reasoning layers for better learning
            'H_cycles': 1,  
            'L_cycles': 1,  
            'H_layers': 2,  # Reduced from 3 for faster convergence
            'L_layers': 2,  # Reduced from 3 for faster convergence
            
            # Optimized model size for board games
            'hidden_size': 128,  # Reduced from 256
            'expansion': 2.0,    
            'num_heads': 8,  # Increased for better attention
            'pos_encodings': 'learned_2d',  # Custom 2D positional encoding
            
            # Minimal ACT (almost no adaptive computation)
            'halt_max_steps': 8,  # Force single step for deterministic output
            'halt_exploration_prob': 0.1,  # No exploration
            
            # Enable move and value prediction for AlphaZero
            'use_move_prediction': True,
            'num_actions': self.action_size,
            'move_prediction_from_token': 0,  # Use first board position
            
            'use_value_prediction': True,
            'value_prediction_from_token': 0,  # Use first board position
            
            'forward_dtype': 'bfloat16',
            
            # Board-specific configuration for 2D encoding
            'board_x': self.board_x,
            'board_y': self.board_y
        }
        
        # Initialize HRM model with bridge
        hrm_model = HierarchicalReasoningModel_ACTV1(hrm_config)
        self.bridge = HRMAlphaZeroBridge(hrm_model, self.action_size, self.board_size)

    def forward(self, s):
        # Use the bridge to handle HRM conversion
        pi, v, q_info = self.bridge(s)
        if self.training:
            return F.log_softmax(pi, dim=1), torch.tanh(v), q_info

        # Return in AlphaZero format: log probabilities and tanh values
        return F.log_softmax(pi, dim=1), torch.tanh(v)
