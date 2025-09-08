#!/usr/bin/env python3
"""
Bridge between AlphaZero training format and HRM model expectations.
Adapts AlphaZero (board, pi, v) training data to HRM's expected format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class HRMAlphaZeroBridge(nn.Module):
    """
    Wrapper that makes HRM model work with AlphaZero training data.
    Converts (board, pi, v) format to HRM's expected batch format.
    """
    
    def __init__(self, hrm_model, action_size, board_size):
        super().__init__()
        self.hrm_model = hrm_model
        self.action_size = action_size
        self.board_size = board_size
        # New chess format: (112, 8, 8)
        self.board_x = self.board_y = 8  # Chess is always 8x8

    def forward(self, boards):
        """
        Forward pass that converts AlphaZero board format to HRM format.
        
        Args:
            boards: For chess: [batch_size, 64, 112] - square encodings
                   For othello: [batch_size, board_x, board_y] - old format
            
        Returns:
            pi: [batch_size, action_size] - Policy logits (NOT log probabilities)
            v: [batch_size, 1] - Value predictions  
        """
        batch_size = boards.shape[0]
        # Chess format: boards is [batch_size, 112, 8, 8]
        # HRM with chess tokenization expects this format directly
        batch = {
            'inputs': boards,  # Use square encodings directly
            'puzzle_identifiers': torch.zeros(batch_size, dtype=torch.long, device=boards.device),
            'move_targets': torch.full((batch_size,), -100, dtype=torch.long, device=boards.device),
            'value_targets': torch.full((batch_size,), float('nan'), dtype=torch.float, device=boards.device),
        }
        
        # Initialize carry state
        carry = self.hrm_model.initial_carry(batch)
        
        # Move to correct device
        if hasattr(carry.inner_carry, 'z_H'):
            carry.inner_carry.z_H = carry.inner_carry.z_H.to(boards.device)
            carry.inner_carry.z_L = carry.inner_carry.z_L.to(boards.device)
        if hasattr(carry, 'steps'):
            carry.steps = carry.steps.to(boards.device)
            carry.halted = carry.halted.to(boards.device)
        for key in carry.current_data:
            carry.current_data[key] = carry.current_data[key].to(boards.device)
        
        # Forward through HRM
        _, outputs = self.hrm_model(carry, batch)
        
        # Extract policy and value
        pi = outputs['move_logits']  # Raw logits for AlphaZero
        v = outputs['value_logits']
        q_info = {}
        q_info["q_halt_logits"] = outputs['q_halt_logits']
        q_info["q_continue_logits"] = outputs['q_continue_logits']
        if "target_q_continue" in outputs:
            q_info["target_q_continue"] = outputs["target_q_continue"]
        return pi, v, q_info

    def train(self, mode=True):
        """Override train to ensure HRM model is also in train mode."""
        super().train(mode)
        self.hrm_model.train(mode)
        return self

    def eval(self):
        """Override eval to ensure HRM model is also in eval mode."""
        super().eval()
        self.hrm_model.eval()
        return self


class SimpleHRMLoss:
    """
    Simplified loss functions that work with HRM outputs.
    Mimics AlphaZero loss but handles HRM's raw logit outputs.
    """
    
    # @staticmethod
    # def loss_pi(targets, outputs):
    #     """
    #     Policy loss for HRM outputs.
    #     targets: probability distribution [batch_size, action_size]
    #     outputs: raw logits [batch_size, action_size]
    #     """
    #     # Convert logits to log probabilities
    #     log_probs = F.log_softmax(outputs, dim=1)
    #     # Cross-entropy with target probabilities
    #     return -torch.mean(torch.sum(targets * log_probs, dim=1))
    
    # @staticmethod
    # def loss_v(targets, outputs):
    #     """
    #     Value loss for HRM outputs.
    #     targets: target values [batch_size]
    #     outputs: predicted values [batch_size, 1]
    #     """
    #     # Apply tanh to raw outputs to get [-1, 1] range
    #     predictions = torch.tanh(outputs.view(-1))
    #     return F.mse_loss(predictions, targets)

    @staticmethod
    def loss_pi(targets, outputs):
        # Use HRM-compatible loss for log probabilities
        # return -torch.mean(torch.sum(targets * outputs, dim=1))

        # Mask illegal moves (marked as -1 in target)
        mask = targets >= 0
        targets = torch.clamp(targets, min=0.0)
        
        # Normalize to valid probability distribution
        targets = targets / (targets.sum(dim=1, keepdim=True) + 1e-8)
        # Set logits for masked (illegal) moves to a large negative value
        masked_outputs = outputs.clone()
        masked_outputs[~mask] = -1e9
        # Cross-entropy with target probabilities
        return F.cross_entropy(masked_outputs, targets, reduction='mean')
    
    @staticmethod
    def loss_v(targets, outputs):
        # Use HRM-compatible loss for tanh values
        return F.cross_entropy(outputs, targets, reduction='mean')

    @staticmethod
    def loss_q_halt(move_targets, move_preds, value_targets, value_preds, q_info):
        """
        Calculate Q-halt loss using the methodology from losses.py.
        Determines sequence correctness based on move and value predictions.
        """
        with torch.no_grad():
            # For Othello: move_targets is policy distribution, move_preds is log probabilities
            # Get the actual move from the target policy (argmax of the policy distribution)
            target_moves = torch.argmax(move_targets, dim=-1)
            predicted_moves = torch.argmax(move_preds, dim=-1)
            
            # Handle value predictions - convert to scalar if needed  
            if value_preds.dim() > 1:
                value_predictions = value_preds.squeeze(-1)
            else:
                value_predictions = value_preds
            
            # Move correctness: compare the predicted move with target move
            move_correct = (predicted_moves == target_moves)

            # Value correctness: exact match between target and prediction
            target_val = torch.argmax(value_targets, dim=-1)
            predicted_val = torch.argmax(value_predictions, dim=-1)
            value_correct = (target_val == predicted_val)  # Exact match for chess WDL
            # Overall sequence correctness (both move and value must be correct)
            seq_is_correct = move_correct & value_correct
            
        # Q-halt loss using binary cross-entropy with logits
        q_halt_logits = q_info["q_halt_logits"]
        if q_halt_logits.dim() > 1:
            q_halt_logits = q_halt_logits.squeeze(-1)
            
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits, 
            seq_is_correct.to(q_halt_logits.dtype), 
            reduction="mean"
        )
        return q_halt_loss

    @staticmethod
    def loss_q_continue(q_info):
        # Simplified binary cross-entropy loss for continue logits
        if "target_q_continue" in q_info:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                q_info["q_continue_logits"],
                q_info["target_q_continue"],
                reduction="mean"
            )
            return q_continue_loss
        else:
            return torch.tensor(0.0, device=q_info["q_continue_logits"].device)