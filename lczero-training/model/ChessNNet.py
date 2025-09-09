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
            'use_chess_tokenization': True,  # Use chess-specific tokenization

            # Optimized reasoning layers for better learning
            'H_cycles': 2,  
            'L_cycles': 2,  
            'H_layers': 2,
            'L_layers': 2,
            
            # Optimized model size for board games
            'hidden_size': 256,  # Reduced hidden size for efficiency
            'expansion': 2.0,    
            'num_heads': 8,  # Increased for better attention
            'pos_encodings': 'rope',  # Custom 2D positional encoding

            # How many steps the model can take
            'halt_max_steps': 1,  # Force single step for deterministic output
            'halt_exploration_prob': 0.0,  # Some exploration
            
            # Enable move and value prediction for AlphaZero
            'use_move_prediction': True,
            'num_actions': self.action_size,
            'move_prediction_from_token': 0,  # Use first board position
            # Enable attention-based policy
            'use_attention_policy': True,  # Use attention policy instead of direct logits
            
            'use_value_prediction': True,
            'value_prediction_from_token': 0,  # Use first board position
            
            'use_moves_left_prediction': True,
            'moves_left_from_token': 0,  # Use first board position
            
            # Enable TensorFlow-style heads for better spatial processing
            'use_tensorflow_style_heads': True,
            'value_embedding_size': 32,
            'moves_embedding_size': 8,
            
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
        pi, v, moves_left, q_info = self.bridge(s)

        # Return in AlphaZero format: log probabilities and tanh values
        return F.log_softmax(pi, dim=1), torch.tanh(v), moves_left, q_info
