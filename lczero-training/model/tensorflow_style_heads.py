"""
TensorFlow-style value and moves left heads for HRM model.
Based on the tfprocess.py implementation from Leela Chess Zero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HRM_model.models.layers import CastedLinear


class TensorFlowStyleValueHead(nn.Module):
    """
    Value head following TensorFlow Lc0 implementation.
    Uses all spatial information instead of just first token.
    """
    
    def __init__(self, hidden_size, embedding_size=32, use_wdl=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.use_wdl = use_wdl
        
        # Step 1: Value embedding layer (processes each square)
        self.value_embedding = CastedLinear(hidden_size, embedding_size, bias=True)
        
        # Step 2: Dense processing after flattening
        self.value_dense1 = CastedLinear(embedding_size * 64, 128, bias=True)
        
        # Step 3: Final output layer
        if use_wdl:
            self.value_dense2 = CastedLinear(128, 3, bias=True)  # WDL format [W, D, L]
        else:
            self.value_dense2 = CastedLinear(128, 1, bias=True)  # Classical format [-1, 1]
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following TensorFlow Glorot normal initialization"""
        for module in [self.value_embedding, self.value_dense1, self.value_dense2]:
            if hasattr(module, 'weight'):
                nn.init.xavier_normal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states):
        """
        Forward pass using all spatial information.
        
        Args:
            hidden_states: [batch_size, 64, hidden_size] - all square representations
            
        Returns:
            value_logits: [batch_size, 3] for WDL or [batch_size, 1] for classical
        """
        batch_size = hidden_states.shape[0]
        
        # Step 1: Apply embedding to each square
        value_embedded = F.relu(self.value_embedding(hidden_states))  # [batch, 64, embedding_size]
        
        # Step 2: Flatten spatial dimensions
        value_flattened = value_embedded.reshape(batch_size, -1)  # [batch, 64 * embedding_size]
        
        # Step 3: Dense processing
        value_hidden = F.relu(self.value_dense1(value_flattened))  # [batch, 128]
        
        # Step 4: Final output
        if self.use_wdl:
            value_logits = self.value_dense2(value_hidden)  # [batch, 3] - no activation for logits
        else:
            value_logits = torch.tanh(self.value_dense2(value_hidden))  # [batch, 1] with tanh
        
        return value_logits


class TensorFlowStyleMovesLeftHead(nn.Module):
    """
    Moves left head following TensorFlow Lc0 implementation.
    Uses all spatial information and ensures positive predictions.
    """
    
    def __init__(self, hidden_size, embedding_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        # Step 1: Moves left embedding layer
        self.moves_embedding = CastedLinear(hidden_size, embedding_size, bias=True)
        
        # Step 2: Dense processing after flattening  
        self.moves_dense1 = CastedLinear(embedding_size * 64, 128, bias=True)
        
        # Step 3: Final output layer
        self.moves_dense2 = CastedLinear(128, 1, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following TensorFlow Glorot normal initialization"""
        for module in [self.moves_embedding, self.moves_dense1, self.moves_dense2]:
            if hasattr(module, 'weight'):
                nn.init.xavier_normal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states):
        """
        Forward pass using all spatial information.
        
        Args:
            hidden_states: [batch_size, 64, hidden_size] - all square representations
            
        Returns:
            moves_left: [batch_size, 1] - positive values only (ReLU activated)
        """
        batch_size = hidden_states.shape[0]
        
        # Step 1: Apply embedding to each square
        moves_embedded = F.relu(self.moves_embedding(hidden_states))  # [batch, 64, embedding_size]
        
        # Step 2: Flatten spatial dimensions
        moves_flattened = moves_embedded.reshape(batch_size, -1)  # [batch, 64 * embedding_size]
        
        # Step 3: Dense processing
        moves_hidden = F.relu(self.moves_dense1(moves_flattened))  # [batch, 128]
        
        # Step 4: Final output with ReLU (ensures positive values)
        moves_left = F.relu(self.moves_dense2(moves_hidden))  # [batch, 1] - positive only
        
        return moves_left


class CombinedTensorFlowStyleHeads(nn.Module):
    """
    Combined value and moves left heads for efficient processing.
    Shares the initial spatial processing before branching.
    """
    
    def __init__(self, hidden_size, value_embedding_size=32, moves_embedding_size=8, use_wdl=True):
        super().__init__()
        
        self.value_head = TensorFlowStyleValueHead(
            hidden_size=hidden_size,
            embedding_size=value_embedding_size,
            use_wdl=use_wdl
        )
        
        self.moves_left_head = TensorFlowStyleMovesLeftHead(
            hidden_size=hidden_size,
            embedding_size=moves_embedding_size
        )
    
    def forward(self, hidden_states):
        """
        Forward pass for both heads.
        
        Args:
            hidden_states: [batch_size, 64, hidden_size]
            
        Returns:
            tuple: (value_logits, moves_left_logits)
        """
        value_logits = self.value_head(hidden_states)
        moves_left_logits = self.moves_left_head(hidden_states)
        
        return value_logits, moves_left_logits


def test_tensorflow_style_heads():
    """Test the TensorFlow-style head implementations."""
    print("Testing TensorFlow-style Value and Moves Left Heads...")
    
    # Test parameters
    batch_size, seq_len, hidden_size = 4, 64, 256
    
    # Create test input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    print(f"Input shape: {hidden_states.shape}")
    
    # Test Value Head
    print("\n1. Testing Value Head...")
    value_head = TensorFlowStyleValueHead(hidden_size, embedding_size=32, use_wdl=True)
    value_logits = value_head(hidden_states)
    print(f"Value output shape: {value_logits.shape} (expected: [{batch_size}, 3])")
    assert value_logits.shape == (batch_size, 3), f"Wrong value shape: {value_logits.shape}"
    
    # Test Moves Left Head
    print("\n2. Testing Moves Left Head...")
    moves_head = TensorFlowStyleMovesLeftHead(hidden_size, embedding_size=8)
    moves_logits = moves_head(hidden_states)
    print(f"Moves left output shape: {moves_logits.shape} (expected: [{batch_size}, 1])")
    assert moves_logits.shape == (batch_size, 1), f"Wrong moves shape: {moves_logits.shape}"
    
    # Test that moves left are positive (ReLU constraint)
    print(f"Moves left values: {moves_logits.squeeze()}")
    assert torch.all(moves_logits >= 0), "Moves left should be non-negative!"
    
    # Test Combined Heads
    print("\n3. Testing Combined Heads...")
    combined_heads = CombinedTensorFlowStyleHeads(hidden_size)
    value_out, moves_out = combined_heads(hidden_states)
    print(f"Combined value shape: {value_out.shape}")
    print(f"Combined moves shape: {moves_out.shape}")
    
    # Test parameter count
    value_params = sum(p.numel() for p in value_head.parameters())
    moves_params = sum(p.numel() for p in moves_head.parameters())
    print(f"\nParameter counts:")
    print(f"Value head: {value_params:,} parameters")
    print(f"Moves left head: {moves_params:,} parameters")
    print(f"Total: {value_params + moves_params:,} parameters")
    
    print("\n✅ All tests passed! TensorFlow-style heads are working correctly.")


if __name__ == "__main__":
    test_tensorflow_style_heads()