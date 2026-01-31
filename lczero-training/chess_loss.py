import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
    
class ChessLoss(nn.Module):
    """Combined loss function for chess training (adapted from tfprocess.py)"""
    
    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0,
                 moves_left_weight: float = 0.01, reg_weight: float = 3e-5, config=None):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.moves_left_weight = moves_left_weight
        self.reg_weight = reg_weight
        self.config = config
        self.model_type = config.get('model_type')
        
    def policy_loss(self, policy_targets: torch.Tensor, policy_outputs: torch.Tensor) -> torch.Tensor:
        """Policy cross-entropy loss with illegal move masking"""
        # Mask illegal moves (marked as -1 in target)
        mask = policy_targets >= 0
        policy_targets_clamped = torch.clamp(policy_targets, min=0.0)

        # Normalize to valid probability distribution
        target_sum = policy_targets_clamped.sum(dim=1, keepdim=True)
        policy_targets_normalized = policy_targets_clamped / (target_sum + 1e-8)

        # Set logits for masked (illegal) moves to a large negative value
        # Use -1e4 instead of -1e9 to avoid numerical issues with float16
        policy_outputs_masked = policy_outputs.clone()
        policy_outputs_masked[~mask] = -1e4

        # Compute log probabilities from logits
        log_probs = F.log_softmax(policy_outputs_masked, dim=1)

        # Cross-entropy: -sum(target * log_prob)
        # Apply mask to zero out illegal moves before multiplication
        log_probs_legal = log_probs * mask.float()
        loss = -torch.mean(torch.sum(policy_targets_normalized * log_probs_legal, dim=1))

        return loss

    def value_loss(self, value_targets: torch.Tensor, value_outputs: torch.Tensor) -> torch.Tensor:
        """WDL value cross-entropy loss"""
        return F.cross_entropy(value_outputs, value_targets, reduction='mean')

    def moves_left_loss(self, ml_targets: torch.Tensor, ml_outputs: torch.Tensor) -> torch.Tensor:
        """Huber loss for moves left prediction"""
        scale = 20.0
        ml_targets_scaled = ml_targets / scale
        ml_outputs_scaled = ml_outputs / scale

        return F.huber_loss(ml_outputs_scaled, ml_targets_scaled, delta=10.0/scale)

    def calculate_move_accuracy(self, policy_targets, policy_outputs):
        """
        Calculate top-1 move prediction accuracy.
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

    def calculate_top3_accuracy(self, policy_targets, policy_outputs):
        """
        Calculate top-3 move prediction accuracy.
        Returns 1 if target move is in top 3 predictions, 0 otherwise.
        """
        with torch.no_grad():
            # Get the target move (highest probability move)
            target_moves = torch.argmax(policy_targets, dim=-1)

            mask = policy_targets >= 0
            move_preds = policy_outputs.clone()
            move_preds[~mask] = -1e9  # Mask illegal moves

            # Get top 3 predicted moves
            _, top3_indices = torch.topk(move_preds, k=3, dim=-1)

            # Check if target is in top 3
            target_moves_expanded = target_moves.unsqueeze(-1)  # [batch, 1]
            correct_predictions = (top3_indices == target_moves_expanded).any(dim=-1)
            accuracy = correct_predictions.float().mean()

            return accuracy
    
    def loss_q_halt(self, policy_targets, policy_outputs, value_targets, value_outputs, q_info):
        """
        Calculate Q-halt loss using the methodology from losses.py.
        Determines sequence correctness based on move and value predictions.
        """
        with torch.no_grad():
            # For Othello: move_targets is policy distribution, move_preds is log probabilities
            # Get the actual move from the target policy (argmax of the policy distribution)
            target_moves = torch.argmax(policy_targets, dim=-1)
            mask = policy_targets >= 0
            policy_outputs = policy_outputs.clone()
            policy_outputs[~mask] = -1e9  # Mask illegal moves
            predicted_moves = torch.argmax(policy_outputs, dim=-1)
            
            # Handle value predictions - convert to scalar if needed  
            if value_outputs.dim() > 1:
                value_predictions = value_outputs.squeeze(-1)
            else:
                value_predictions = value_outputs
            
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

    def loss_q_continue(self, q_info):
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

    def forward(self, policy_targets: torch.Tensor, policy_outputs: torch.Tensor,
                value_targets: torch.Tensor, value_outputs: torch.Tensor,
                ml_targets: torch.Tensor, ml_outputs: torch.Tensor, q_info: dict,
                model: nn.Module) -> Tuple[torch.Tensor, dict]:
        """Combined loss calculation"""

        # Individual losses
        p_loss = self.policy_loss(policy_targets, policy_outputs)
        v_loss = self.value_loss(value_targets, value_outputs)
        ml_loss = self.moves_left_loss(ml_targets, ml_outputs)
        # L2 regularization
        reg_loss = sum(p.pow(2.0).sum() for p in model.parameters()) * self.reg_weight
        p_accuracy = self.calculate_move_accuracy(policy_targets, policy_outputs)
        p_top3_accuracy = self.calculate_top3_accuracy(policy_targets, policy_outputs)
        # Combined loss
        total_loss = (self.policy_weight * p_loss +
                     self.value_weight * v_loss +
                     self.moves_left_weight * ml_loss +
                     reg_loss)

        # add in q losses for hrm/trm, if the halt_max_steps is set > 1
        if self.config.get('model_type') == "hrm":
            if self.config.get('hrm_config').get('halt_max_steps') > 1:
                q_halt_loss = self.loss_q_halt(policy_targets, policy_outputs, value_targets, value_outputs, q_info)
                # Check if use_q_continue is enabled (default: True for backwards compatibility)
                no_ACT_continue = self.config.get('hrm_config').get('no_ACT_continue', True)
                if not no_ACT_continue:
                    q_continue_loss = self.loss_q_continue(q_info)
                    q_loss = 0.5 * (q_halt_loss + q_continue_loss)
                else:
                    q_continue_loss = torch.tensor(0.0, device=q_halt_loss.device)
                    q_loss = q_halt_loss
                total_loss += q_loss
        elif self.config.get('model_type') == "trm":
            if self.config.get('trm_config').get('halt_max_steps') > 1:
                q_halt_loss = self.loss_q_halt(policy_targets, policy_outputs, value_targets, value_outputs, q_info)
                # Check if use_q_continue is enabled (default: True for backwards compatibility)
                no_ACT_continue = self.config.get('trm_config').get('no_ACT_continue', True)
                if not no_ACT_continue:
                    q_continue_loss = self.loss_q_continue(q_info)
                    q_loss = 0.5 * (q_halt_loss + q_continue_loss)
                else:
                    q_continue_loss = torch.tensor(0.0, device=q_halt_loss.device)
                    q_loss = q_halt_loss
                total_loss += q_loss

        # Return loss components for logging
        loss_dict = {
            'policy_loss': p_loss.item(),
            'value_loss': v_loss.item(),
            'moves_left_loss': ml_loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item(),
            'policy_accuracy': p_accuracy.item(),
            'policy_top3_accuracy': p_top3_accuracy.item(),
            'q_halt_loss': 0.0,
            'q_continue_loss': 0.0,
            'q_loss': 0.0
        }

        if self.config.get('model_type') == "hrm":
            if self.config.get('hrm_config').get('halt_max_steps') > 1:
                loss_dict['q_halt_loss'] = q_halt_loss.item()
                loss_dict['q_continue_loss'] = q_continue_loss.item()
                loss_dict['q_loss'] = q_loss.item()
        elif self.config.get('model_type') == "trm":
            if self.config.get('trm_config').get('halt_max_steps') > 1:
                loss_dict['q_halt_loss'] = q_halt_loss.item()
                loss_dict['q_continue_loss'] = q_continue_loss.item()
                loss_dict['q_loss'] = q_loss.item()

        return total_loss, loss_dict