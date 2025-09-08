from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
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

    # Move prediction config (NEW)
    use_move_prediction: bool = False
    num_actions: int = 0  # Number of possible moves/actions
    move_prediction_from_token: int = 0  # Which token position to use for move prediction
    
    # Value prediction config (NEW)
    use_value_prediction: bool = False
    value_prediction_from_token: int = 0  # Which token position to use for value prediction
    
    # Chess tokenization config (NEW)
    use_chess_tokenization: bool = False
    square_feature_dim: int = 112  # Features per square (historical + game state)

    forward_dtype: str = "bfloat16"


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

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
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)
        
        # Move prediction head (NEW)
        if self.config.use_move_prediction:
            self.move_head = CastedLinear(self.config.hidden_size, self.config.num_actions, bias=True)
            # Initialize move head with small weights for stable learning
            with torch.no_grad():
                self.move_head.weight.normal_(0, 0.01)
                if self.move_head.bias is not None:
                    self.move_head.bias.zero_()
        
        # Value prediction head (NEW)
        if self.config.use_value_prediction:
            self.value_head = CastedLinear(self.config.hidden_size, 1, bias=True)  # Single value output
            # Initialize value head with small weights for stable learning
            with torch.no_grad():
                self.value_head.weight.normal_(0, 0.01)
                if self.value_head.bias is not None:
                    self.value_head.bias.zero_()

        # Chess square tokenization (NEW)
        if self.config.use_chess_tokenization:
            # Linear projection from square features to hidden dimension
            self.square_projection = CastedLinear(self.config.square_feature_dim, self.config.hidden_size, bias=True)
            
            # Per-square positional encodings (learned offset and scale vectors)
            self.square_pos_offsets = nn.Parameter(torch.zeros(self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype))
            self.square_pos_scales = nn.Parameter(torch.ones(self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype))
            
            # Initialize square projection with proper scaling
            with torch.no_grad():
                # Xavier/Glorot initialization for the projection
                std = (2.0 / (self.config.square_feature_dim + self.config.hidden_size)) ** 0.5
                self.square_projection.weight.normal_(0, std)
                if self.square_projection.bias is not None:
                    self.square_projection.bias.zero_()
                
                # Small random initialization for positional encodings
                self.square_pos_offsets.normal_(0, 0.02)
                self.square_pos_scales.normal_(1.0, 0.02)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])
        
        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Handle different input formats:
        if input.dim() == 3 and self.config.use_chess_tokenization:  
            # [batch, squares, features] - proper chess square tokenization
            batch_size, num_squares, features_per_square = input.shape
            
            # Step 1: Linear projection for each square
            # Reshape to [batch * squares, features] for projection
            input_flat = input.view(batch_size * num_squares, features_per_square).to(self.forward_dtype)
            
            # Apply linear projection: [batch * squares, features] -> [batch * squares, hidden_size]
            projected = self.square_projection(input_flat)
            
            # Reshape back to [batch, squares, hidden_size]
            embedding = projected.view(batch_size, num_squares, self.config.hidden_size)
            
            # Step 2: Add per-square positional encodings
            # Each square gets its own learned offset and scale
            pos_offsets = self.square_pos_offsets.unsqueeze(0)  # [1, squares, hidden_size]
            pos_scales = self.square_pos_scales.unsqueeze(0)   # [1, squares, hidden_size]
            
            # Apply positional encoding: embedding * scale + offset
            embedding = embedding * pos_scales + pos_offsets
            
        elif input.dim() == 3:  # [batch, squares, features] - fallback to sum tokenization
            batch_size, num_squares, features_per_square = input.shape
            # Each square has multiple features that need to be converted to a single token
            # Use a hash/sum of the features as the token index
            token_indices = torch.clamp(input.sum(dim=-1), 0, self.config.vocab_size - 1)
            # Now token_indices is [batch, squares] where each square becomes one token
            embedding = self.embed_tokens(token_indices.to(torch.int32))
            
        elif input.dim() == 2:  # Original 1D format [batch, seq_len] - direct token indices
            embedding = self.embed_tokens(input.to(torch.int32))
        else:
            raise ValueError(f"Unsupported input tensor dimension: {input.dim()}")

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings (only for non-chess tokenization)
        if self.config.pos_encodings == "learned" and not self.config.use_chess_tokenization:
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale (only for non-chess tokenization)
        if not self.config.use_chess_tokenization:
            embedding = self.embed_scale * embedding
            
        return embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        # Move prediction outputs (NEW)
        move_logits = None
        if self.config.use_move_prediction:
            # Use specified token position for move prediction (default: first token after puzzle embedding)
            move_token_idx = self.puzzle_emb_len + self.config.move_prediction_from_token
            move_logits = self.move_head(z_H[:, move_token_idx]).to(torch.float32)
        
        # Value prediction outputs (NEW)
        value_logits = None
        if self.config.use_value_prediction:
            # Use specified token position for value prediction (default: first token after puzzle embedding)
            value_token_idx = self.puzzle_emb_len + self.config.value_prediction_from_token
            value_logits = self.value_head(z_H[:, value_token_idx]).squeeze(-1).to(torch.float32)  # Remove last dimension
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), move_logits, value_logits


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        inner_result = self.inner(new_inner_carry, new_current_data)
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = inner_result[:3]
        move_logits = inner_result[3] if len(inner_result) > 3 else None
        value_logits = inner_result[4] if len(inner_result) > 4 else None

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        # Add move prediction outputs (NEW)
        if move_logits is not None:
            outputs["move_logits"] = move_logits
        
        # Add value prediction outputs (NEW)
        if value_logits is not None:
            outputs["value_logits"] = value_logits
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_inner_result = self.inner(new_inner_carry, new_current_data)
                next_q_halt_logits, next_q_continue_logits = next_inner_result[2]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
