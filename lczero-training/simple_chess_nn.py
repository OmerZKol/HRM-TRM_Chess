import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SimpleChessNet(nn.Module):
    """Simple test architecture for chess training"""
    
    def __init__(self, input_channels=112, hidden_dim=256, policy_size=1858):
        super().__init__()
        
        # Convolutional layers for spatial features
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_size)
        
        # Value head  
        self.value_fc1 = nn.Linear(256, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, 3)  # WDL output
        
        # Moves left head
        self.moves_left_fc1 = nn.Linear(256, hidden_dim // 2)
        self.moves_left_fc2 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        # x shape: (batch_size, 112, 8, 8)
        
        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))
        
        # Policy head - keep spatial dimensions
        policy_features = F.relu(self.policy_conv(x))
        policy_flat = policy_features.view(policy_features.size(0), -1)
        policy_logits = self.policy_fc(policy_flat)
        
        # Value and moves left heads - use global pooling
        global_features = self.global_pool(x).view(x.size(0), -1)
        
        # Value head (WDL)
        value_hidden = F.relu(self.value_fc1(global_features))
        value_logits = self.value_fc2(value_hidden)
        
        # Moves left head
        moves_hidden = F.relu(self.moves_left_fc1(global_features))
        moves_left = self.moves_left_fc2(moves_hidden)
        
        return policy_logits, value_logits, moves_left, {}
    
class ChessLoss(nn.Module):
    """Combined loss function for chess training (adapted from tfprocess.py)"""
    
    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0, 
                 moves_left_weight: float = 0.01, reg_weight: float = 1e-4, model_type: str = "simple"):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.moves_left_weight = moves_left_weight
        self.reg_weight = reg_weight
        self.model_type = model_type
        
    def policy_loss(self, policy_targets: torch.Tensor, policy_outputs: torch.Tensor) -> torch.Tensor:
        """Policy cross-entropy loss with illegal move masking"""
        # Mask illegal moves (marked as -1 in target)
        mask = policy_targets >= 0
        policy_targets = torch.clamp(policy_targets, min=0.0)

        # Normalize to valid probability distribution
        policy_targets = policy_targets / (policy_targets.sum(dim=1, keepdim=True) + 1e-8)
        # Set logits for masked (illegal) moves to a large negative value
        policy_outputs = policy_outputs.clone()
        policy_outputs[~mask] = -1e9
        # Cross-entropy with target probabilities
        return F.cross_entropy(policy_outputs, policy_targets, reduction='mean')

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
        Calculate move prediction accuracy.
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
        # Combined loss
        total_loss = (self.policy_weight * p_loss + 
                     self.value_weight * v_loss + 
                     self.moves_left_weight * ml_loss + 
                     reg_loss)
        
        # add in q losses for hrm
        if(self.model_type == "hrm"):
            q_halt_loss = self.loss_q_halt(policy_targets, policy_outputs, value_targets, value_outputs, q_info)
            q_continue_loss = self.loss_q_continue(q_info)
            q_loss = 0.5 * (q_halt_loss + q_continue_loss)
            total_loss += q_loss
        
        # Return loss components for logging
        loss_dict = {
            'policy_loss': p_loss.item(),
            'value_loss': v_loss.item(), 
            'moves_left_loss': ml_loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item(),
            'policy_accuracy': p_accuracy.item(),
            'q_halt_loss': 0.0,
            'q_continue_loss': 0.0,
            'q_loss': 0.0
        }

        if(self.model_type == "hrm"):
            loss_dict['q_halt_loss'] = q_halt_loss.item()
            loss_dict['q_continue_loss'] = q_continue_loss.item()
            loss_dict['q_loss'] = q_loss.item()
            
        return total_loss, loss_dict