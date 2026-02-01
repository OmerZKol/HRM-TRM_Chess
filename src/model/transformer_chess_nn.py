"""
Transformer-based chess model that accepts board input and returns action, policy, and moves left heads.
Uses implementations from the HRM_Chess/lczero-training/model directory to avoid redundant code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common.layers import (
    Attention, SwiGLU, RotaryEmbedding, CastedLinear, CastedEmbedding, rms_norm
)
from model.heads.attention_policy import AttentionPolicyHead
from model.heads.value_heads import CombinedTensorFlowStyleHeads


class ChessBoardEmbedding(nn.Module):
    """
    Converts chess board representation to transformer input embeddings.
    Handles the 112-channel board representation from LCZero format.
    Supports both additive and concatenative (arc_encoding) positional encoding styles.
    """

    def __init__(self, square_feature_dim=112, hidden_size=256, max_seq_len=64,
                 arc_encoding=False, pos_enc_dim=16):
        super().__init__()
        self.square_feature_dim = square_feature_dim
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.arc_encoding = arc_encoding
        self.pos_enc_dim = pos_enc_dim

        if arc_encoding:
            # TensorFlow/TRM style: concatenate positional encoding with features
            self.pos_enc = nn.Parameter(
                torch.randn(64, pos_enc_dim) * 0.02
            )
            # Project concatenated features: [square_feature_dim + pos_enc_dim] -> hidden_size
            self.square_projection = CastedLinear(
                square_feature_dim + pos_enc_dim, hidden_size, bias=True
            )
            # Gating mechanism (multiplicative + additive)
            from model.common.gating import ma_gating
            self.embedding_activation = nn.ReLU()
            self.input_gate = ma_gating(64, hidden_size)
        else:
            # Original transformer style: additive positional embeddings
            # Convert spatial board to sequence of square embeddings
            self.square_projection = CastedLinear(square_feature_dim, hidden_size, bias=True)

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
        board_flat = board_tensor.view(batch_size, self.square_feature_dim, 64)

        # Transpose to get features per square: [batch, 64, 112]
        square_features = board_flat.transpose(1, 2)

        if self.arc_encoding:
            # TRM style: concatenate positional encoding with features
            positional_encoding = self.pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 64, pos_enc_dim]

            # Concatenate: [batch, 64, 112] + [batch, 64, 16] -> [batch, 64, 128]
            concatenated_input = torch.cat([square_features, positional_encoding], dim=2)

            # Project to hidden size: [batch, 64, hidden_size]
            embeddings = self.square_projection(concatenated_input)

            # Apply activation
            embeddings = self.embedding_activation(embeddings)

            # Apply gating
            embeddings = self.input_gate(embeddings)
        else:
            # Original transformer style: additive positional embeddings
            # Project to hidden size: [batch, 64, hidden_size]
            square_embeddings = self.square_projection(square_features)

            # Add positional embeddings
            positions = torch.arange(64, device=board_tensor.device)
            pos_embeddings = self.position_embedding(positions)  # [64, hidden_size]

            # Add positional embeddings (broadcasting)
            embeddings = square_embeddings + pos_embeddings.unsqueeze(0)

            # Apply layer normalization
            embeddings = self.norm(embeddings)

        positions = torch.arange(64, device=board_tensor.device)
        return embeddings, positions

class TransformerBlock(nn.Module):
    """
    Single transformer block with attention and feed-forward layers.
    """
    
    def __init__(self, hidden_size, num_heads, expansion=4.0, rms_norm_eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Pre-norm for attention
        self.attention_norm_eps = rms_norm_eps
        
        # Multi-head attention - use fallback for CPU
        self.attention = Attention(
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,  # Full attention, no GQA
            causal=False  # Chess is not causal - all squares can attend to each other
        )
        
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
    
    def __init__(self, config=None, board_size=None):
        """
        Initialize TransformerChessNet.

        Args:
            config: Configuration dictionary with 'transformer_config' section
            board_size: Board dimensions (for compatibility with HRM/TRM) - must be (8,8)
        """
        super().__init__()

        # Validate board_size for compatibility
        if board_size is not None and board_size != (8, 8):
            raise ValueError(f"Only (8, 8) board size is supported, got {board_size}")

        # Extract transformer config or use defaults
        if config is not None:
            transformer_config = config.get('transformer_config', {})
            policy_size = 1858
        else:
            # Fallback to defaults if no config provided (for backward compatibility)
            transformer_config = {}
            policy_size = 1858

        # Extract parameters from config
        square_feature_dim = transformer_config.get('square_feature_dim', 112)
        hidden_size = transformer_config.get('hidden_size', 512)
        num_layers = transformer_config.get('num_layers', 4)
        num_heads = transformer_config.get('num_heads', 8)
        expansion = transformer_config.get('expansion', 4.0)
        max_position_embeddings = transformer_config.get('max_position_embeddings', 64)
        rope_base = transformer_config.get('rope_base', 10000)
        use_wdl = transformer_config.get('use_wdl', True)
        rms_norm_eps = transformer_config.get('rms_norm_eps', 1e-5)

        # Input embedding configuration (TRM-style arc_encoding)
        arc_encoding = transformer_config.get('arc_encoding', False)
        pos_enc_dim = transformer_config.get('pos_enc_dim', 16)

        # Policy head configuration
        use_attention_policy = transformer_config.get('use_attention_policy', True)
        value_embedding_size = transformer_config.get('value_embedding_size', 32)
        moves_embedding_size = transformer_config.get('moves_embedding_size', 8)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.policy_size = policy_size
        self.use_attention_policy = use_attention_policy

        # Board embedding
        self.board_embedding = ChessBoardEmbedding(
            square_feature_dim=square_feature_dim,
            hidden_size=hidden_size,
            max_seq_len=max_position_embeddings,
            arc_encoding=arc_encoding,
            pos_enc_dim=pos_enc_dim
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
                rms_norm_eps=rms_norm_eps
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

        # Output heads - choose policy head type
        if use_attention_policy:
            # Attention-based policy head (like Leela Chess Zero)
            self.policy_head = AttentionPolicyHead(
                hidden_size=hidden_size,
                num_heads=num_heads
            )
        else:
            # Direct linear policy head (simpler, faster)
            self.policy_head = CastedLinear(hidden_size, policy_size, bias=True)
            # Initialize with small weights for stable training
            with torch.no_grad():
                self.policy_head.weight.normal_(0, 0.1)
                if self.policy_head.bias is not None:
                    self.policy_head.bias.zero_()

        # Value and moves left heads
        self.value_moves_heads = CombinedTensorFlowStyleHeads(
            hidden_size=hidden_size,
            value_embedding_size=value_embedding_size,
            moves_embedding_size=moves_embedding_size,
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

        # Policy head - handle both attention-based and direct linear
        if self.use_attention_policy:
            # Attention-based policy: returns logits and attention weights
            policy_logits, policy_attention = self.policy_head(hidden_states_f32)
        else:
            # Direct linear policy: use first token (like value head)
            policy_logits = self.policy_head(hidden_states_f32[:, 0, :])  # [batch, policy_size]
            policy_attention = None  # No attention weights for direct policy

        # Value and moves left heads (use all spatial information)
        value_logits, moves_left = self.value_moves_heads(hidden_states_f32)

        # Additional info for analysis/debugging
        info_dict = {
            'policy_attention': policy_attention,  # [batch, 64, 64] or None
            'hidden_states': hidden_states,        # [batch, 64, hidden_size]
            'use_attention_policy': self.use_attention_policy
        }

        return policy_logits, value_logits, moves_left, info_dict


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params