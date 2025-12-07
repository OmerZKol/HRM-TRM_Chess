from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from model.trm_model.common import trunc_normal_init_
from model.trm_model.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from model.trm_model.sparse_embedding import CastedSparseEmbedding

# Chess-specific imports (optional, only used when chess features enabled)
try:
    from model.attention_policy_map import AttentionPolicyHead
    from model.tensorflow_style_heads import TensorFlowStyleValueHead, TensorFlowStyleMovesLeftHead
except ImportError:
    AttentionPolicyHead = None
    TensorFlowStyleValueHead = None
    TensorFlowStyleMovesLeftHead = None

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    # Chess always uses 64 squares
    seq_len: int = 64

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

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

    # Board dimensions
    board_x: int = 8
    board_y: int = 8

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

    forward_dtype: str = "bfloat16"

class Gating(nn.Module):
    def __init__(self, hidden_size: int, additive: bool = True, init_value: float = 0.0):
        super().__init__()
        self.additive = additive
        self.gate = nn.Parameter(torch.full((hidden_size,), init_value))
        if not additive:
            self.gate.data.clamp_(min=0) # Equivalent to NonNeg constraint

    def forward(self, x):
        if self.additive:
            return x + self.gate
        else:
            # Equivalent to NonNeg constraint during training
            if self.training:
                # Use non-inplace operation to avoid breaking gradients
                gate = torch.clamp(self.gate, min=0)
            else:
                gate = self.gate
            return x * gate

def ma_gating(hidden_size: int):
    return nn.Sequential(
        Gating(hidden_size, additive=False, init_value=1.0),
        Gating(hidden_size, additive=True, init_value=0.0)
    )

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config, layer_idx: int = 0, total_layers: int = 1) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
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

        # Residual scaling for deep networks (helps gradient flow)
        # Scale decreases with depth to prevent gradient explosion
        self.residual_scale = (total_layers / (layer_idx + 1)) ** 0.5 if total_layers > 1 else 1.0

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Pre-Norm (better gradient flow for deep networks)
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(rms_norm(hidden_states, variance_epsilon=self.norm_eps))
            hidden_states = hidden_states + self.residual_scale * out
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention with residual scaling
            normed = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
            if torch.isnan(normed).any():
                print(f"[Block NaN] NaN in rms_norm before attention")
                print(f"  hidden_states stats: min={hidden_states.min():.4f}, max={hidden_states.max():.4f}, mean={hidden_states.mean():.4f}")
                raise RuntimeError("NaN in rms_norm before attention")

            attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=normed)
            if torch.isnan(attn_out).any():
                print(f"[Block NaN] NaN in attention output")
                raise RuntimeError("NaN in attention output")

            scaled_attn = self.residual_scale * attn_out
            if torch.isnan(scaled_attn).any():
                print(f"[Block NaN] NaN after residual scaling attention (scale={self.residual_scale:.4f})")
                print(f"  attn_out stats: min={attn_out.min():.4f}, max={attn_out.max():.4f}, mean={attn_out.mean():.4f}")
                raise RuntimeError("NaN after residual scaling attention")

            hidden_states = hidden_states + scaled_attn
            if torch.isnan(hidden_states).any():
                print(f"[Block NaN] NaN after adding attention residual")
                raise RuntimeError("NaN after adding attention residual")

        # Fully Connected with residual scaling
        normed = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        if torch.isnan(normed).any():
            print(f"[Block NaN] NaN in rms_norm before MLP")
            print(f"  hidden_states stats: min={hidden_states.min():.4f}, max={hidden_states.max():.4f}, mean={hidden_states.mean():.4f}")
            raise RuntimeError("NaN in rms_norm before MLP")

        out = self.mlp(normed)
        if torch.isnan(out).any():
            print(f"[Block NaN] NaN in MLP output")
            raise RuntimeError("NaN in MLP output")

        scaled_mlp = self.residual_scale * out
        if torch.isnan(scaled_mlp).any():
            print(f"[Block NaN] NaN after residual scaling MLP (scale={self.residual_scale:.4f})")
            print(f"  mlp_out stats: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
            raise RuntimeError("NaN after residual scaling MLP")

        hidden_states = hidden_states + scaled_mlp
        if torch.isnan(hidden_states).any():
            print(f"[Block NaN] NaN after adding MLP residual")
            raise RuntimeError("NaN after adding MLP residual")

        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block], norm_eps: float = 1e-5):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.norm_eps = norm_eps

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection

        # NaN detection: Check input injection result
        if torch.isnan(hidden_states).any():
            print(f"[L_level NaN] NaN after input_injection addition")
            print(f"  hidden_states stats before: min={hidden_states.min():.4f}, max={hidden_states.max():.4f}")
            print(f"  input_injection stats: min={input_injection.min():.4f}, max={input_injection.max():.4f}")

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
            # NaN detection: Check after each layer
            if torch.isnan(hidden_states).any():
                print(f"[L_level NaN] NaN detected after layer {layer_idx}/{len(self.layers)}")
                raise RuntimeError(f"NaN in L_level layer {layer_idx}")

        # Final norm (standard for pre-norm architectures)
        hidden_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)

        # NaN detection: Check after final norm
        if torch.isnan(hidden_states).any():
            print(f"[L_level NaN] NaN detected after final rms_norm")
            raise RuntimeError(f"NaN in L_level after final rms_norm")

        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Q head for halting
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Chess square tokenization
        if self.config.arc_encoding:
            # positional encoding concatenated with features
            self.pos_enc = nn.Parameter(torch.randn(self.config.board_x * self.config.board_y, self.config.pos_enc_dim, dtype=self.forward_dtype) * 0.02)
            self.square_projection = CastedLinear(self.config.square_feature_dim + self.config.pos_enc_dim, self.config.hidden_size, bias=True)
            self.input_gate = ma_gating(self.config.hidden_size)
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

        # Reasoning Layers with layer indices for residual scaling
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config, layer_idx=i, total_layers=self.config.L_layers) for i in range(self.config.L_layers)],
            norm_eps=self.config.rms_norm_eps
        )

        # Initial states (reduced std for better convergence with bfloat16)
        # Using std=0.02 similar to modern transformer initializations
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02), persistent=True)

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
                # Use TensorFlow-style value head that processes all spatial information
                if TensorFlowStyleValueHead is None:
                    raise ImportError("TensorFlowStyleValueHead not available. Check imports.")
                self.value_head = TensorFlowStyleValueHead(
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
                # Use TensorFlow-style moves left head that processes all spatial information
                if TensorFlowStyleMovesLeftHead is None:
                    raise ImportError("TensorFlowStyleMovesLeftHead not available. Check imports.")
                self.moves_left_head = TensorFlowStyleMovesLeftHead(
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

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
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

        return embedding

    def empty_carry(self, batch_size: int):
        # Always 64 squares for chess
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, 64, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, 64, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L

        # NaN detection in forward pass
        if torch.isnan(z_H).any():
            print(f"[TRM NaN Detection] z_H has NaN before forward iterations")
        if torch.isnan(z_L).any():
            print(f"[TRM NaN Detection] z_L has NaN before forward iterations")
        if torch.isnan(input_embeddings).any():
            print(f"[TRM NaN Detection] input_embeddings has NaN")

        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_H_plus_input = z_H + input_embeddings
                    if torch.isnan(z_H_plus_input).any():
                        print(f"[TRM NaN Detection] z_H + input_embeddings produced NaN (no_grad)")
                    z_L = self.L_level(z_L, z_H_plus_input, **seq_info)
                    if torch.isnan(z_L).any():
                        print(f"[TRM NaN Detection] z_L has NaN after L_level (no_grad)")
                z_H = self.L_level(z_H, z_L, **seq_info)
                if torch.isnan(z_H).any():
                    print(f"[TRM NaN Detection] z_H has NaN after L_level (no_grad)")
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_H_plus_input = z_H + input_embeddings
            if torch.isnan(z_H_plus_input).any():
                print(f"[TRM NaN Detection] z_H + input_embeddings produced NaN (with grad)")
                print(f"  z_H stats: min={z_H.min():.4f}, max={z_H.max():.4f}, mean={z_H.mean():.4f}")
                print(f"  input_embeddings stats: min={input_embeddings.min():.4f}, max={input_embeddings.max():.4f}, mean={input_embeddings.mean():.4f}")
            z_L = self.L_level(z_L, z_H_plus_input, **seq_info)
            if torch.isnan(z_L).any():
                print(f"[TRM NaN Detection] z_L has NaN after L_level (with grad)")
        z_H = self.L_level(z_H, z_L, **seq_info)
        if torch.isnan(z_H).any():
            print(f"[TRM NaN Detection] z_H has NaN after final L_level")

        # Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad

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

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

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
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, (next_q_halt_logits, next_q_continue_logits), _, _, _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
