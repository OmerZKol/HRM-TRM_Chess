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
        
        # Forward through HRM with looping until all sequences halt
        # Track which sequences have completed (halted at least once)
        has_completed = torch.zeros(batch_size, dtype=torch.bool, device=boards.device)

        # Safety limit: halt_max_steps + 1 for initial halted state + 1 extra for safety
        max_iterations = self.hrm_model.config.halt_max_steps + 2
        c = 0
        for iteration in range(max_iterations):
            carry, outputs = self.hrm_model(carry, batch)
            print(c)
            c+=1
            # Mark sequences that have halted (but haven't halted in previous iterations due to reset)
            # A sequence completes when it reaches a halted state
            has_completed = has_completed | carry.halted

            # Exit when all sequences have completed at least once
            if has_completed.all():
                break
        else:
            # This should never happen - log a warning
            print(f"Warning: HRM loop reached max iterations ({max_iterations}) without all sequences halting")
            print(f"Halted status: {carry.halted}")
            print(f"Has completed: {has_completed}")
            print(f"Steps: {carry.steps if hasattr(carry, 'steps') else 'N/A'}")

        # Extract policy and value
        pi = outputs['move_logits']  # Raw logits for AlphaZero
        v = outputs['value_logits']
        q_info = {}
        q_info["q_halt_logits"] = outputs['q_halt_logits']
        q_info["q_continue_logits"] = outputs['q_continue_logits']
        if "target_q_continue" in outputs:
            q_info["target_q_continue"] = outputs["target_q_continue"]
        # Add recursion steps for tracking
        if "recursion_steps" in outputs:
            q_info["recursion_steps"] = outputs["recursion_steps"]
        moves_left = outputs["moves_left_logits"]
        return pi, v, moves_left, q_info

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
            mask = move_targets >= 0
            move_preds = move_preds.clone()
            move_preds[~mask] = -1e9  # Mask illegal moves
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

    @staticmethod
    def loss_ml(target_moves_left, pred_moves_left):
        """Huber loss for moves left prediction"""
        scale = 20.0
        ml_target_scaled = target_moves_left / scale
        ml_output_scaled = pred_moves_left / scale
        
        return F.huber_loss(ml_output_scaled, ml_target_scaled, delta=10.0/scale)

    @staticmethod
    def calculate_move_accuracy(policy_targets, policy_outputs):
        """
        Calculate move prediction accuracy.
        
        Args:
            policy_targets: Target policy distribution [batch_size, action_size]  
            policy_outputs: Predicted policy logits [batch_size, action_size]
            
        Returns:
            accuracy: Fraction of correctly predicted moves
        """
        with torch.no_grad():
            # Get the target move (highest probability move)
            target_moves = torch.argmax(policy_targets, dim=-1)

            mask = policy_targets >= 0
            move_preds = policy_outputs.clone()

            move_preds[~mask] = -1e9  # Mask illegal moves
            # Get the predicted move (highest logit move)  
            predicted_moves = torch.argmax(move_preds, dim=-1)

            # Calculate accuracy
            correct_predictions = (predicted_moves == target_moves)
            accuracy = correct_predictions.float().mean()
            
            return accuracy
        