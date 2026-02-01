import torch
import torch.nn as nn
import torch.nn.functional as F

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
