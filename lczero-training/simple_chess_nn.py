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
        
        return policy_logits, value_logits, moves_left
    
class ChessLoss(nn.Module):
    """Combined loss function for chess training (adapted from tfprocess.py)"""
    
    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0, 
                 moves_left_weight: float = 0.01, reg_weight: float = 1e-4):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.moves_left_weight = moves_left_weight
        self.reg_weight = reg_weight
        
    def policy_loss(self, policy_target: torch.Tensor, policy_output: torch.Tensor) -> torch.Tensor:
        """Policy cross-entropy loss with illegal move masking"""
        # Mask illegal moves (marked as -1 in target)
        mask = policy_target >= 0
        policy_target = torch.clamp(policy_target, min=0.0)
        
        # Normalize to valid probability distribution
        policy_target = policy_target / (policy_target.sum(dim=1, keepdim=True) + 1e-8)
        
        return F.cross_entropy(policy_output, policy_target, reduction='mean')
    
    def value_loss(self, value_target: torch.Tensor, value_output: torch.Tensor) -> torch.Tensor:
        """WDL value cross-entropy loss"""
        return F.cross_entropy(value_output, value_target, reduction='mean')
    
    def moves_left_loss(self, ml_target: torch.Tensor, ml_output: torch.Tensor) -> torch.Tensor:
        """Huber loss for moves left prediction"""
        scale = 20.0
        ml_target_scaled = ml_target / scale
        ml_output_scaled = ml_output / scale
        
        return F.huber_loss(ml_output_scaled, ml_target_scaled, delta=10.0/scale)
    
    def forward(self, policy_target: torch.Tensor, policy_output: torch.Tensor,
                value_target: torch.Tensor, value_output: torch.Tensor,
                ml_target: torch.Tensor, ml_output: torch.Tensor,
                model: nn.Module) -> Tuple[torch.Tensor, dict]:
        """Combined loss calculation"""
        
        # Individual losses
        p_loss = self.policy_loss(policy_target, policy_output)
        v_loss = self.value_loss(value_target, value_output)
        ml_loss = self.moves_left_loss(ml_target, ml_output)
        
        # L2 regularization
        reg_loss = sum(p.pow(2.0).sum() for p in model.parameters()) * self.reg_weight
        
        # Combined loss
        total_loss = (self.policy_weight * p_loss + 
                     self.value_weight * v_loss + 
                     self.moves_left_weight * ml_loss + 
                     reg_loss)
        
        # Return loss components for logging
        loss_dict = {
            'policy_loss': p_loss.item(),
            'value_loss': v_loss.item(), 
            'moves_left_loss': ml_loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict