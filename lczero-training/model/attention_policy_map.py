import numpy as np
import torch
import torch.nn as nn

# Chess move vectors (from original attention_policy_map.py)
move = np.arange(1, 8)

diag = np.array([
    move    + move*8,
    move    - move*8,
    move*-1 - move*8,
    move*-1 + move*8
])

orthog = np.array([
    move,
    move*-8,
    move*-1,
    move*8
])

knight = np.array([
    [2 + 1*8],
    [2 - 1*8],
    [1 - 2*8],
    [-1 - 2*8],
    [-2 - 1*8],
    [-2 + 1*8],
    [-1 + 2*8],
    [1 + 2*8]
])

promos = np.array([2*8, 3*8, 4*8])
pawn_promotion = np.array([
    -1 + promos,
    0 + promos,
    1 + promos
])


def make_attention_policy_map():
    """
    Creates mapping matrix from attention weights to chess moves.
    Returns: [4288, 1858] sparse matrix mapping attention indices to UCI moves
    """
    # Generate all theoretically possible moves from each square
    traversable = []
    for i in range(8):  # ranks
        for j in range(8):  # files
            sq = (8*i + j)
            traversable.append(
                sq +
                np.sort(
                    np.int32(
                        np.concatenate((
                            orthog[0][:7-j], orthog[2][:j], orthog[1][:i], orthog[3][:7-i],
                            diag[0][:np.min((7-i, 7-j))], diag[3][:np.min((7-i, j))],
                            diag[1][:np.min((i, 7-j))], diag[2][:np.min((i, j))],
                            knight[0] if i < 7 and j < 6 else [], knight[1] if i > 0 and j < 6 else [],
                            knight[2] if i > 1 and j < 7 else [], knight[3] if i > 1 and j > 0 else [],
                            knight[4] if i > 0 and j > 1 else [], knight[5] if i < 7 and j > 1 else [],
                            knight[6] if i < 6 and j > 0 else [], knight[7] if i < 6 and j < 7 else [],
                            pawn_promotion[0] if i == 6 and j > 0 else [],
                            pawn_promotion[1] if i == 6           else [],
                            pawn_promotion[2] if i == 6 and j < 7 else [],
                        ))
                    )
                )
            )

    # Create sparse mapping matrix
    z = np.zeros((64*64+8*24, 1858), dtype=np.float32)
    
    # Standard moves (64x64 attention -> moves)
    i = 0
    for pickup_index, putdown_indices in enumerate(traversable):
        for putdown_index in putdown_indices:
            if putdown_index < 64:
                z[putdown_index + (64*pickup_index), i] = 1
                i += 1
    
    # Promotion moves (8x24 promotion attention -> promotion moves)
    j = 0
    j1 = np.array([3, -2, 3, -2, 3])
    j2 = np.array([3, 3, -5, 3, 3, -5, 3, 3, 1])
    ls = np.append(j1, 1)
    for k in range(6):
        ls = np.append(ls, j2)
    ls = np.append(ls, j1)
    ls = np.append(ls, 0)
    
    for pickup_index, putdown_indices in enumerate(traversable):
        for putdown_index in putdown_indices:
            if putdown_index >= 64:
                pickup_file = pickup_index % 8
                promotion_file = putdown_index % 8
                promotion_rank = (putdown_index // 8) - 8
                z[4096 + pickup_file*24 + (promotion_file*3+promotion_rank), i] = 1
                i += ls[j]
                j += 1

    return z


class AttentionPolicyHead(nn.Module):
    """
    Attention-based policy head that interprets attention weights as chess moves.
    """
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Query and Key projections for policy attention
        self.policy_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.policy_k = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Promotion offset prediction
        self.promotion_keys = nn.Linear(hidden_size, hidden_size, bias=False)
        self.promotion_offsets = nn.Linear(hidden_size, 4, bias=False)
        
        # Attention policy mapping matrix
        mapping_matrix = make_attention_policy_map()
        self.register_buffer('policy_map', torch.from_numpy(mapping_matrix))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following original tfprocess.py"""
        nn.init.xavier_normal_(self.policy_q.weight)
        nn.init.xavier_normal_(self.policy_k.weight)
        nn.init.xavier_normal_(self.promotion_keys.weight)
        nn.init.xavier_normal_(self.promotion_offsets.weight)
    
    def forward(self, hidden_states):
        """
        Convert hidden states to policy logits via attention.
        
        Args:
            hidden_states: [batch_size, 64, hidden_size] - one per chess square
            
        Returns:
            policy_logits: [batch_size, 1858] - logits for all chess moves
            attention_weights: [batch_size, 64, 64] - for visualization/analysis
        """
        batch_size = hidden_states.shape[0]
        
        # Generate queries and keys for each square
        queries = self.policy_q(hidden_states)  # [batch, 64, hidden_size]
        keys = self.policy_k(hidden_states)     # [batch, 64, hidden_size]
        
        # Multi-head attention computation
        queries = queries.view(batch_size, 64, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, 64, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention: [batch, heads, 64, 64]
        dk = torch.tensor(self.head_dim, dtype=torch.float32, device=queries.device)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(dk)
        
        # Average across heads to get [batch, 64, 64] attention matrix
        policy_attention = attention_scores.mean(dim=1)
        
        # Handle pawn promotions
        promotion_logits = self._compute_promotion_logits(hidden_states, policy_attention)
        
        # Scale attention scores
        policy_attention_scaled = policy_attention / torch.sqrt(dk)
        
        # Flatten attention weights and promotion logits
        flattened_attention = policy_attention_scaled.reshape(batch_size, -1)  # [batch, 4096]
        flattened_promotions = promotion_logits.reshape(batch_size, -1)        # [batch, 192]
        
        # Concatenate: [batch, 4288] = [batch, 4096+192]
        combined = torch.cat([flattened_attention, flattened_promotions], dim=1)
        
        # Apply policy mapping: [batch, 4288] @ [4288, 1858] = [batch, 1858]
        policy_logits = torch.matmul(combined, self.policy_map)
        
        return policy_logits, policy_attention
    
    def _compute_promotion_logits(self, hidden_states, policy_attention):
        """
        Compute promotion-specific logits for pawn promotion moves.
        
        Args:
            hidden_states: [batch, 64, hidden_size]
            policy_attention: [batch, 64, 64]
            
        Returns:
            promotion_logits: [batch, 8, 24] - 8 files × 24 promotion moves per file
        """
        batch_size = hidden_states.shape[0]
        
        # Get promotion keys from 7th rank squares (squares 48-55)
        promotion_squares = hidden_states[:, 48:56, :]  # [batch, 8, hidden_size]
        promotion_keys = self.promotion_keys(promotion_squares)  # [batch, 8, hidden_size]
        
        # Generate promotion offsets (Q, R, B relative to default Knight)
        promotion_offsets = self.promotion_offsets(promotion_keys)  # [batch, 8, 4]
        promotion_offsets = promotion_offsets.transpose(-2, -1)  # [batch, 4, 8]
        
        # Base promotion logits from attention (7th rank to 8th rank)
        # Squares 48-55 (7th rank) to 56-63 (8th rank)
        base_promo_logits = policy_attention[:, 48:56, 56:64]  # [batch, 8, 8]
        
        # Apply offsets for Q, R, B (Knight is default/base)
        dk = torch.tensor(self.head_dim, dtype=torch.float32, device=promotion_offsets.device)
        scaled_offsets = promotion_offsets * torch.sqrt(dk)
        
        # Create promotion logits for each piece type
        q_promo = (base_promo_logits + scaled_offsets[:, 0:1, :]).unsqueeze(-1)  # [batch, 8, 8, 1]
        r_promo = (base_promo_logits + scaled_offsets[:, 1:2, :]).unsqueeze(-1)  # [batch, 8, 8, 1]
        b_promo = (base_promo_logits + scaled_offsets[:, 2:3, :]).unsqueeze(-1)  # [batch, 8, 8, 1]
        
        # Concatenate: [batch, 8, 8, 3]
        promotion_logits = torch.cat([q_promo, r_promo, b_promo], dim=-1)
        
        # Reshape to [batch, 8, 24] (8 files × 3 piece types × 8 target squares)
        return promotion_logits.reshape(batch_size, 8, 24)


def test_attention_policy_map():
    """Test the attention policy mapping"""
    print("Testing Attention Policy Map...")
    
    # Test mapping matrix creation
    mapping = make_attention_policy_map()
    print(f"Mapping matrix shape: {mapping.shape}")
    print(f"Mapping matrix non-zero elements: {np.count_nonzero(mapping)}")
    
    # Test attention policy head
    batch_size, seq_len, hidden_size, num_heads = 2, 64, 256, 8
    
    policy_head = AttentionPolicyHead(hidden_size, num_heads)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    policy_logits, attention_weights = policy_head(hidden_states)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print("Test passed!")


if __name__ == "__main__":
    test_attention_policy_map()