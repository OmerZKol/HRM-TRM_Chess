"""
Transformer Baseline for Architecture Ablation

This is an architecture ablation of the Hierarchical Reasoning Model (HRM).
Key changes from the full HRM (trm.py):
1. REMOVED hierarchical split (no separate H and L levels) - single z_H only
2. REMOVED inner cycles (no H_cycles/L_cycles loops within reasoning) - single pass
3. KEPT ACT outer loop structure intact
4. KEPT all data preprocessing, embeddings, and chess-specific heads

Architecture: Single-level transformer that processes the 64-square chess board
with the same positional encodings and chess heads as the full HRM.
"""

from typing import Tuple, List, Dict
from dataclasses import dataclass

import torch
import copy
from torch import nn
from pydantic import BaseModel
import random
from model.common.initialization import trunc_normal_init_
from model.common.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedLinear
from model.common.gating import Gating, ma_gating

# Chess-specific imports (optional, only used when chess features enabled)
try:
    from model.heads.attention_policy import AttentionPolicyHead
    from model.heads.value_heads import ValueHead, MovesLeftHead
except ImportError:
    AttentionPolicyHead = None
    ValueHead = None
    MovesLeftHead = None

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor  # Single level only (no z_L in baseline)


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    # Chess always uses 64 squares
    seq_len: int = 64

    # Baseline: no inner cycles, these are kept for config compatibility but ignored
    H_cycles: int = 1  # Ignored in baseline
    L_cycles: int = 1  # Ignored in baseline

    H_layers: int  # Number of transformer layers (used in baseline)
    L_layers: int = 0  # Ignored in baseline

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    # Move prediction config
    use_move_prediction: bool = True
    num_actions: int = 1858  # Number of possible chess moves
    move_prediction_from_token: int = 0  # Which token position to use for move prediction

    # Value prediction config
    use_value_prediction: bool = True
    value_prediction_from_token: int = 0  # Which token position to use for value prediction

    use_moves_left_prediction: bool = True
    moves_left_from_token: int = 0  # Which token position to use for moves left prediction

    # Chess tokenization config (always enabled)
    square_feature_dim: int = 112  # Features per square (historical + game state)
    
    # From tfprocess.py
    arc_encoding: bool = True
    pos_enc_dim: int = 16

    # Policy head type
    use_attention_policy: bool = False  # Use attention-based policy instead of direct

    # Head architecture types
    use_tensorflow_style_heads: bool = False  # Use TensorFlow-style value/moves heads
    value_embedding_size: int = 32  # Value head embedding size
    moves_embedding_size: int = 8   # Moves left head embedding size

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    no_ACT_continue: bool = True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    # ACT control flags (baseline additions)
    act_enabled: bool = True  # If False, always run halt_max_steps (no early stopping during training)
    act_inference: bool = False  # If True, use adaptive computation during inference

    forward_dtype: str = "bfloat16"

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post-Norm: normalize AFTER residual addition to bound activation magnitudes
        # This is critical for numerical stability with weight sharing and multiple cycles
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention with post-norm
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps
            )

        # MLP with post-norm
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers (post-norm blocks already bound output magnitudes, no final norm needed)
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Embedding scale to match original model magnitude
        # Chess input is sparse binary (mostly 0/1), so projected embeddings have small magnitude (~0.2-0.3)
        # This scale brings input embeddings to comparable magnitude with hidden states (~1.0)
        import math
        self.embed_scale = math.sqrt(self.config.hidden_size)

        # Q head for halting
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Chess square tokenization
        if self.config.arc_encoding:
            # positional encoding concatenated with features
            self.pos_enc = nn.Parameter(torch.randn(64, self.config.pos_enc_dim, dtype=self.forward_dtype) * 0.02)
            self.square_projection = CastedLinear(self.config.square_feature_dim + self.config.pos_enc_dim, self.config.hidden_size, bias=True)
            self.embedding_activation = nn.ReLU()
            self.input_gate = ma_gating(64, self.config.hidden_size)
        else:
            # learned per-square offsets and scales
            self.square_projection = CastedLinear(self.config.square_feature_dim, self.config.hidden_size, bias=True)
            # Per-square positional encodings (learned offset and scale vectors)
            self.square_pos_offsets = nn.Parameter(torch.zeros(self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype))
            self.square_pos_scales = nn.Parameter(torch.ones(self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype))

        # Initialize square projection with proper scaling
        with torch.no_grad():
            # Xavier/Glorot initialization for the projection
            in_dim = self.square_projection.weight.shape[1]
            std = (2.0 / (in_dim + self.config.hidden_size)) ** 0.5
            self.square_projection.weight.normal_(0, std)
            if self.square_projection.bias is not None:
                self.square_projection.bias.zero_()

            if not self.config.arc_encoding:
                # Small random initialization for positional encodings
                self.square_pos_offsets.normal_(0, 0.02)
                self.square_pos_scales.normal_(1.0, 0.02)

        # RoPE for attention (optional, based on config)
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=64,
                base=self.config.rope_theta
            )

        # Reasoning Layers - single level (baseline: uses H_layers, not L_layers)
        self.H_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _ in range(self.config.H_layers)]
        )

        # Initial state (single level only - no L_init in baseline)
        # Post-norm architecture bounds outputs, so std=1.0 is stable
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1.0), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

        # Chess-specific heads
        if self.config.use_move_prediction:
            if self.config.use_attention_policy:
                # Use attention-based policy head
                if AttentionPolicyHead is None:
                    raise ImportError("AttentionPolicyHead not available. Check imports.")
                self.move_head = AttentionPolicyHead(self.config.hidden_size, self.config.num_heads)
            else:
                # Use direct policy head
                self.move_head = CastedLinear(self.config.hidden_size, self.config.num_actions, bias=True)
                # AlphaZero-style initialization for policy head
                with torch.no_grad():
                    self.move_head.weight.normal_(0, 0.1)
                    if self.move_head.bias is not None:
                        self.move_head.bias.zero_()

        # Value prediction head
        if self.config.use_value_prediction:
            if self.config.use_tensorflow_style_heads:
                # Use value head that processes all spatial information
                if ValueHead is None:
                    raise ImportError("ValueHead not available. Check imports.")
                self.value_head = ValueHead(
                    hidden_size=self.config.hidden_size,
                    embedding_size=self.config.value_embedding_size,
                    use_wdl=True
                )
            else:
                # Original single-token value head
                # 3 for WDL style output
                self.value_head = CastedLinear(self.config.hidden_size, 3, bias=True)
                # AlphaZero-style initialization for value head
                with torch.no_grad():
                    self.value_head.weight.normal_(0, 0.1)
                    if self.value_head.bias is not None:
                        self.value_head.bias.zero_()

        # Moves left prediction head
        if self.config.use_moves_left_prediction:
            if self.config.use_tensorflow_style_heads:
                # Use moves left head that processes all spatial information
                if MovesLeftHead is None:
                    raise ImportError("MovesLeftHead not available. Check imports.")
                self.moves_left_head = MovesLeftHead(
                    hidden_size=self.config.hidden_size,
                    embedding_size=self.config.moves_embedding_size
                )
            else:
                # Original single-token moves left head
                self.moves_left_head = CastedLinear(self.config.hidden_size, 1, bias=True)
                with torch.no_grad():
                    self.moves_left_head.weight.normal_(0, 0.1)
                    if self.moves_left_head.bias is not None:
                        self.moves_left_head.bias.zero_()

    def _input_embeddings(self, input: torch.Tensor):
        """
        Convert chess board input to embeddings.
        Input format: [batch, features, height, width] -> [batch, 64, hidden_size]
        """
        # Reshape input: [batch, features, 8, 8] -> [batch, 64, features]
        original_shape = input.shape
        input = input.permute(0, 2, 3, 1)  # [batch, 8, 8, features]
        input = input.reshape(-1, 64, original_shape[1])  # [batch, 64, features]

        batch_size, num_squares, _ = input.shape

        if self.config.arc_encoding:
            # tfprocess.py style: concatenate positional encoding with features
            # self.pos_enc is [64, pos_enc_dim]
            positional_encoding = self.pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 64, pos_enc_dim]

            # Concatenate: [batch, 64, 112] + [batch, 64, 16] -> [batch, 64, 128]
            concatenated_input = torch.cat([input.to(self.forward_dtype), positional_encoding], dim=2)

            # Linear projection: [batch, 64, 128] -> [batch, 64, hidden_size]
            input_flat = concatenated_input.view(batch_size * num_squares, -1)
            projected = self.square_projection(input_flat)
            embedding = projected.view(batch_size, num_squares, self.config.hidden_size)

            embedding = self.embedding_activation(embedding)

            # Apply gating (gating includes activation internally)
            embedding = self.input_gate(embedding)
        else:
            # Original HRM implementation: per-square offsets and scales
            # Linear projection: [batch * 64, features] -> [batch * 64, hidden_size]
            input_flat = input.view(batch_size * num_squares, -1).to(self.forward_dtype)
            projected = self.square_projection(input_flat)
            embedding = projected.view(batch_size, num_squares, self.config.hidden_size)

            # Add per-square positional encodings
            pos_offsets = self.square_pos_offsets.unsqueeze(0)  # [1, 64, hidden_size]
            pos_scales = self.square_pos_scales.unsqueeze(0)   # [1, 64, hidden_size]
            embedding = embedding * pos_scales + pos_offsets

        # Scale embeddings to match original model magnitude (~1.0)
        # This is critical for proper gradient flow with ACT loops
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        # Always 64 squares for chess - single level only in baseline
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, 64, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"])

        # Baseline: Single forward pass through H_level (no inner cycles)
        z_H = self.H_level(carry.z_H, input_embeddings, **seq_info)

        # Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach())  # New carry no grad

        # Q head (use first square a1)
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        # Move prediction outputs
        move_logits = None
        attention_weights = None
        if self.config.use_move_prediction:
            if self.config.use_attention_policy:
                # Use attention-based policy: pass all square representations
                move_logits, attention_weights = self.move_head(z_H)
                move_logits = move_logits.to(torch.float32)
            else:
                # Use direct policy: use specified token position
                move_token_idx = self.config.move_prediction_from_token
                move_logits = self.move_head(z_H[:, move_token_idx]).to(torch.float32)

        # Value prediction outputs
        value_logits = None
        if self.config.use_value_prediction:
            if self.config.use_tensorflow_style_heads:
                # TensorFlow-style: pass all spatial information
                value_logits = self.value_head(z_H).to(torch.float32)  # [batch, 3] WDL format
            else:
                # Original: use specified token position for value prediction
                value_token_idx = self.config.value_prediction_from_token
                value_logits = self.value_head(z_H[:, value_token_idx]).to(torch.float32)

        # Moves left prediction outputs
        moves_left_logits = None
        if self.config.use_moves_left_prediction:
            if self.config.use_tensorflow_style_heads:
                # TensorFlow-style: pass all spatial information
                moves_left_logits = self.moves_left_head(z_H).to(torch.float32)  # [batch, 1]
            else:
                # Original: use specified token position for moves left prediction
                moves_left_token_idx = self.config.moves_left_from_token
                moves_left_logits = self.moves_left_head(z_H[:, moves_left_token_idx]).to(torch.float32)

        return new_carry, (q_logits[..., 0], q_logits[..., 1]), move_logits, value_logits, moves_left_logits, attention_weights


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, (q_halt_logits, q_continue_logits), move_logits, value_logits, moves_left_logits, attention_weights = self.inner(new_inner_carry, new_current_data)

        # Build outputs dictionary
        outputs = {
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "move_logits": move_logits,
            "value_logits": value_logits,
            "moves_left_logits": moves_left_logits,
        }

        # Add attention weights if using attention policy
        if attention_weights is not None:
            outputs["attention_weights"] = attention_weights

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            # Add recursion steps to outputs for tracking
            outputs["recursion_steps"] = new_steps.clone()
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # ACT halting logic
            # Use act_enabled flag for training, act_inference flag for eval
            act_active = (self.training and self.config.act_enabled) or (not self.training and self.config.act_inference)
            
            if act_active and (self.config.halt_max_steps > 1):

                # Halt signal

                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                # Force halt if reached max steps (override exploration constraints)
                halted = halted | is_last_step

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, (next_q_halt_logits, next_q_continue_logits), _, _, _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
