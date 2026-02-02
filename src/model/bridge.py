#!/usr/bin/env python3
"""
Bridge between AlphaZero training format and HRM/TRM model expectations.
Adapts AlphaZero (board, pi, v) training data to model's expected format.
"""

import torch
import torch.nn as nn

class AlphaZeroBridge(nn.Module):
    """
    Wrapper for HRM/TRM models, mainly for the ACT recursive feature.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Data format: (112, 8, 8)

    def forward(self, boards):
        """
        Forward pass that converts AlphaZero board format to model format.

        Args:
            boards: For chess: [batch_size, 112, 8, 8] - square encodings

        Returns:
            pi: [batch_size, action_size] - Policy logits (NOT log probabilities)
            v: [batch_size, 3] - Value predictions
            moves_left: [batch_size, 1] - Moves left predictions
            q_info: Dict containing Q-learning information
        """
        batch_size = boards.shape[0]
        # Chess format: boards is [batch_size, 112, 8, 8]
        # Model with chess tokenization expects this format directly
        batch = {
            'inputs': boards,  # Use square encodings directly
        }
        # Initialize carry state
        carry = self.model.initial_carry(batch)

        # Move to correct device - handle both HRM (z_H, z_L) and TRM (z_H only)
        if hasattr(carry.inner_carry, 'z_H'):
            carry.inner_carry.z_H = carry.inner_carry.z_H.to(boards.device)
        if hasattr(carry.inner_carry, 'z_L'):
            carry.inner_carry.z_L = carry.inner_carry.z_L.to(boards.device)
        if hasattr(carry, 'steps'):
            carry.steps = carry.steps.to(boards.device)
            carry.halted = carry.halted.to(boards.device)
        for key in carry.current_data:
            carry.current_data[key] = carry.current_data[key].to(boards.device)

        # Forward through model with looping until all sequences halt
        # Track which sequences have completed (halted at least once)
        has_completed = torch.zeros(batch_size, dtype=torch.bool, device=boards.device)

        # Safety limit: halt_max_steps + 1 for initial halted state + 1 extra for safety
        max_iterations = self.model.config.halt_max_steps + 2
        for _ in range(max_iterations):
            carry, outputs = self.model(carry, batch)
            # Mark sequences that have halted (but haven't halted in previous iterations due to reset)
            # A sequence completes when it reaches a halted state
            has_completed = has_completed | carry.halted

            # Exit when all sequences have completed at least once
            if has_completed.all():
                break
        else:
            # This should never happen - log a warning
            print(f"Warning: Model loop reached max iterations ({max_iterations}) without all sequences halting")
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
        """Override train to ensure model is also in train mode."""
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        """Override eval to ensure model is also in eval mode."""
        super().eval()
        self.model.eval()
        return self
