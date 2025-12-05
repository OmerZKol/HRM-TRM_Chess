"""
Transformer-based chess model that accepts board input and returns action, policy, and moves left heads.
Uses implementations from the HRM_Chess/lczero-training/model directory to avoid redundant code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from model.HRM_model.layers import (
    Attention, SwiGLU, RotaryEmbedding, CastedLinear, CastedEmbedding, rms_norm
)
from model.attention_policy_map import AttentionPolicyHead
from model.tensorflow_style_heads import CombinedTensorFlowStyleHeads


class ChessBoardEmbedding(nn.Module):
    """
    Converts chess board representation to transformer input embeddings.
    Handles the 112-channel board representation from LCZero format.
    """
    
    def __init__(self, input_channels=112, hidden_size=256, max_seq_len=64):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # Convert spatial board to sequence of square embeddings
        self.square_projection = CastedLinear(input_channels, hidden_size, bias=True)
        
        # Positional embeddings for each square (0-63)
        self.position_embedding = CastedEmbedding(
            num_embeddings=64,
            embedding_dim=hidden_size,
            init_std=0.02,
            cast_to=torch.float32  # Will be converted to model dtype later
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, board_tensor):
        """
        Convert board tensor to sequence embeddings.
        
        Args:
            board_tensor: [batch_size, 112, 8, 8] - LCZero board representation
            
        Returns:
            embeddings: [batch_size, 64, hidden_size] - sequence of square embeddings
            positions: [64] - position indices for RoPE
        """
        batch_size = board_tensor.shape[0]
        
        # Flatten spatial dimensions: [batch, 112, 64]
        board_flat = board_tensor.view(batch_size, self.input_channels, 64)
        
        # Transpose to get features per square: [batch, 64, 112]
        square_features = board_flat.transpose(1, 2)
        
        # Project to hidden size: [batch, 64, hidden_size]
        square_embeddings = self.square_projection(square_features)
        
        # Add positional embeddings
        positions = torch.arange(64, device=board_tensor.device)
        pos_embeddings = self.position_embedding(positions)  # [64, hidden_size]
        
        # Add positional embeddings (broadcasting)
        embeddings = square_embeddings + pos_embeddings.unsqueeze(0)
        
        # Apply layer normalization
        embeddings = self.norm(embeddings)
        
        return embeddings, positions


class StandardAttention(nn.Module):
    """Standard PyTorch attention as fallback for CPU testing."""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv_proj = CastedLinear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        
    def forward(self, cos_sin, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get Q, K, V
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE if provided
        if cos_sin is not None:
            from model.HRM_model.models.layers import apply_rotary_pos_emb
            cos, sin = cos_sin
            query = query.transpose(1, 2)  # [batch, seq, heads, head_dim]
            key = key.transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            query = query.transpose(1, 2)  # back to [batch, heads, seq, head_dim] 
            key = key.transpose(1, 2)
        
        # Standard attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    """
    Single transformer block with attention and feed-forward layers.
    """
    
    def __init__(self, hidden_size, num_heads, expansion=4.0, rms_norm_eps=1e-5, use_flash_attn=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Pre-norm for attention
        self.attention_norm_eps = rms_norm_eps
        
        # Multi-head attention - use fallback for CPU
        if use_flash_attn and torch.cuda.is_available():
            self.attention = Attention(
                hidden_size=hidden_size,
                head_dim=self.head_dim,
                num_heads=num_heads,
                num_key_value_heads=num_heads,  # Full attention, no GQA
                causal=False  # Chess is not causal - all squares can attend to each other
            )
        else:
            self.attention = StandardAttention(hidden_size, num_heads)
        
        # Pre-norm for feed-forward
        self.ffn_norm_eps = rms_norm_eps
        
        # Feed-forward network with SwiGLU activation
        self.feed_forward = SwiGLU(hidden_size, expansion)
        
    def forward(self, hidden_states, cos_sin=None):
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            cos_sin: Optional rotary position embeddings
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        # Pre-norm attention with residual connection
        normed = rms_norm(hidden_states, self.attention_norm_eps)
        attn_out = self.attention(cos_sin, normed)
        hidden_states = hidden_states + attn_out
        
        # Pre-norm feed-forward with residual connection
        normed = rms_norm(hidden_states, self.ffn_norm_eps)
        ffn_out = self.feed_forward(normed)
        hidden_states = hidden_states + ffn_out
        
        return hidden_states


class TransformerChessNet(nn.Module):
    """
    Transformer-based chess neural network.
    
    Architecture:
    1. Board embedding: Converts 112-channel 8x8 board to 64 square embeddings
    2. Transformer layers: Self-attention across all squares
    3. Output heads: Policy (attention-based), value (WDL), moves left
    """
    
    def __init__(
        self,
        board_size=None,  # For compatibility with other models - (8,8) 
        policy_size=1858,
        input_channels=112,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        expansion=4.0,
        max_position_embeddings=64,
        rope_base=10000,
        use_wdl=True,
        rms_norm_eps=1e-5
    ):
        super().__init__()
        
        # Validate board_size for compatibility
        if board_size is not None and board_size != (8, 8):
            raise ValueError(f"Only (8, 8) board size is supported, got {board_size}")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.policy_size = policy_size
        
        # Board embedding
        self.board_embedding = ChessBoardEmbedding(
            input_channels=input_channels,
            hidden_size=hidden_size,
            max_seq_len=max_position_embeddings
        )
        
        # Rotary position embeddings
        self.rope = RotaryEmbedding(
            dim=hidden_size // num_heads,
            max_position_embeddings=max_position_embeddings,
            base=rope_base
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                expansion=expansion,
                rms_norm_eps=rms_norm_eps,
                use_flash_attn=True  # Will fallback to standard attention on CPU
            )
            for _ in range(num_layers)
        ])
        
        # Set dtype based on device capabilities  
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16  # Use BFloat16 for flash attention on GPU
        else:
            self.dtype = torch.float32   # Use Float32 for CPU
            
        # Final normalization
        self.final_norm_eps = rms_norm_eps
        
        # Output heads
        self.policy_head = AttentionPolicyHead(
            hidden_size=hidden_size,
            num_heads=num_heads
        )
        
        self.value_moves_heads = CombinedTensorFlowStyleHeads(
            hidden_size=hidden_size,
            value_embedding_size=32,
            moves_embedding_size=8,
            use_wdl=use_wdl
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following best practices."""
        if isinstance(module, CastedLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, CastedEmbedding):
            torch.nn.init.normal_(module.embedding_weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, board_tensor):
        """
        Forward pass through transformer chess model.
        
        Args:
            board_tensor: [batch_size, 112, 8, 8] - LCZero board representation
            
        Returns:
            policy_logits: [batch_size, 1858] - move probabilities
            value_logits: [batch_size, 3] - WDL probabilities  
            moves_left: [batch_size, 1] - estimated moves remaining
            info_dict: dict with additional information (attention weights, etc.)
        """
        # Convert board to sequence embeddings (keep input as float32 initially)
        hidden_states, positions = self.board_embedding(board_tensor)
        
        # Convert to model dtype after embeddings
        hidden_states = hidden_states.to(self.dtype)
        
        # Get rotary position embeddings
        cos, sin = self.rope()
        cos_sin = (cos, sin)
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin)
        
        # Final normalization
        hidden_states = rms_norm(hidden_states, self.final_norm_eps)
        
        # Convert back to float32 for output heads (which use standard nn.Linear)
        hidden_states_f32 = hidden_states.to(torch.float32)
        
        # Policy head (uses attention mechanism)
        policy_logits, policy_attention = self.policy_head(hidden_states_f32)
        
        # Value and moves left heads (use all spatial information)
        value_logits, moves_left = self.value_moves_heads(hidden_states_f32)
        
        # Additional info for analysis/debugging
        info_dict = {
            'policy_attention': policy_attention,  # [batch, 64, 64]
            'hidden_states': hidden_states,        # [batch, 64, hidden_size]
        }
        
        return policy_logits, value_logits, moves_left, info_dict


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_transformer_chess_net():
    """Test the transformer chess model."""
    print("Testing Transformer Chess Model...")
    
    # Model configuration
    batch_size = 4
    input_channels = 112
    hidden_size = 512
    num_layers = 6
    num_heads = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create model
    model = TransformerChessNet(
        input_channels=input_channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        expansion=4.0,
        policy_size=1858,
        use_wdl=True
    ).to(device)
    
    # Create test input (random board)
    board_input = torch.randn(batch_size, input_channels, 8, 8).to(device)
    print(f"Input shape: {board_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        policy_logits, value_logits, moves_left, info_dict = model(board_input)
    
    # Check output shapes
    print(f"Policy logits shape: {policy_logits.shape} (expected: [{batch_size}, 1858])")
    print(f"Value logits shape: {value_logits.shape} (expected: [{batch_size}, 3])")
    print(f"Moves left shape: {moves_left.shape} (expected: [{batch_size}, 1])")
    
    # Check additional info
    print(f"Policy attention shape: {info_dict['policy_attention'].shape}")
    print(f"Hidden states shape: {info_dict['hidden_states'].shape}")
    
    # Parameter count
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    
    # Check that outputs are reasonable
    assert policy_logits.shape == (batch_size, 1858), f"Wrong policy shape: {policy_logits.shape}"
    assert value_logits.shape == (batch_size, 3), f"Wrong value shape: {value_logits.shape}"
    assert moves_left.shape == (batch_size, 1), f"Wrong moves shape: {moves_left.shape}"
    assert torch.all(moves_left >= 0), "Moves left should be non-negative"
    
    print("\n✅ All tests passed! Transformer chess model is working correctly.")
    
    return model


if __name__ == "__main__":
    model = test_transformer_chess_net()