# Chess Model Architecture

This directory contains implementations of Hierarchical Reasoning Models (HRM) and Tiny Recursive Models (TRM) for chess, adapted to work with AlphaZero-style training.

## Directory Structure

```
model/
├── __init__.py                    # Main package exports
├── README.md                      # This file
│
├── common/                        # Shared utilities
│   ├── __init__.py
│   ├── initialization.py          # trunc_normal_init_ function
│   ├── layers.py                  # Transformer layers (Attention, SwiGLU, etc.)
│   ├── sparse_embedding.py        # Sparse embedding implementations
│   └── ema.py                     # Exponential moving average
│
├── hrm/                           # Hierarchical Reasoning Model
│   ├── __init__.py
│   └── hrm_model.py               # Full HRM with H/L hierarchical levels
│
├── trm/                           # Tiny Recursive Models
│   ├── __init__.py
│   ├── trm_model.py               # Full TRM with recursive reasoning
│   └── trm_baseline.py            # Baseline (single-level transformer)
│
├── heads/                         # Output head implementations
│   ├── __init__.py
│   ├── attention_policy.py        # Attention-based policy head
│   └── value_heads.py             # TensorFlow-style value/moves heads
│
├── bridge.py                      # AlphaZero training adapter
├── utils.py                       # Utility classes (AverageMeter, dotdict)
│
└── Training wrappers (backward compatibility):
    ├── ChessNNet.py               # HRM wrapper
    ├── ChessTRMNet.py             # TRM wrapper
    └── ChessTRMBaselineNet.py     # TRM baseline wrapper
```

## Model Architectures

### 1. Hierarchical Reasoning Model (HRM)

**File**: `hrm/hrm_model.py`

**Key Features**:
- **Hierarchical Structure**: Separate high-level (H) and low-level (L) reasoning modules
- **Nested Cycles**: Inner L-cycles and outer H-cycles for multi-level reasoning
- **Adaptive Computation Time (ACT)**: Dynamic halting based on Q-learning
- **Chess Tokenization**: 64-square board with learned positional encodings

**Architecture**:
```
Input (112×8×8) → Square Embeddings (64×hidden_size)
                   ↓
    ┌──────────────────────────────────────┐
    │  H-Level (high-level reasoning)      │
    │  ├─ Transformer Layers (H_layers)    │
    │  └─ Output: z_H                      │
    └──────────────────────────────────────┘
                   ↓ (input injection)
    ┌──────────────────────────────────────┐
    │  L-Level (low-level reasoning)       │
    │  ├─ Transformer Layers (L_layers)    │
    │  └─ Output: z_L                      │
    └──────────────────────────────────────┘
                   ↓
    ┌──────────────────────────────────────┐
    │  Output Heads                        │
    │  ├─ Policy Head (move prediction)    │
    │  ├─ Value Head (WDL prediction)      │
    │  ├─ Moves Left Head                  │
    │  └─ Q-Halt Head (ACT control)        │
    └──────────────────────────────────────┘
```

**Usage**:
```python
from model.hrm import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config

config = HierarchicalReasoningModel_ACTV1Config(
    seq_len=64,
    H_cycles=2,
    L_cycles=3,
    H_layers=4,
    L_layers=4,
    hidden_size=512,
    expansion=2.0,
    num_heads=8,
    pos_encodings="rope",
    halt_max_steps=5,
    num_actions=1858,
)

model = HierarchicalReasoningModel_ACTV1(config.__dict__)
```

### 2. Tiny Recursive Model (TRM)

**File**: `trm/trm_model.py`

**Key Features**:
- Similar hierarchical structure to HRM
- Optimized for smaller model sizes
- Supports LinearSwish activation layers
- Same ACT and chess tokenization as HRM

**Differences from HRM**:
- More compact layer implementation
- Additional LinearSwish layers for expressiveness
- Better suited for resource-constrained training

### 3. TRM Baseline (Single-Level Transformer)

**File**: `trm/trm_baseline.py`

**Key Features**:
- **Ablation Study**: Removes hierarchical H/L split
- **Single-Level**: Only one transformer stack (z_H only)
- **No Inner Cycles**: Single forward pass per ACT step
- **Identical Heads**: Same output heads as HRM/TRM

**Purpose**: Architecture ablation to measure the contribution of hierarchical reasoning.

**Architecture**:
```
Input (112×8×8) → Square Embeddings (64×hidden_size)
                   ↓
    ┌──────────────────────────────────────┐
    │  Single Transformer Stack            │
    │  ├─ Transformer Layers (layers)      │
    │  └─ Output: z_H                      │
    └──────────────────────────────────────┘
                   ↓
    [Same output heads as HRM]
```

## Training Wrappers

For backward compatibility with existing training scripts, use the wrapper classes:

```python
# HRM
from model.ChessNNet import ChessNNet
model = ChessNNet(config)

# TRM
from model.ChessTRMNet import ChessTRMNet
model = ChessTRMNet(config)

# TRM Baseline
from model.ChessTRMBaselineNet import ChessTRMBaselineNet
model = ChessTRMBaselineNet(config)
```

These wrappers:
1. Instantiate the appropriate model
2. Wrap it in `AlphaZeroBridge` for training compatibility
3. Return raw WDL logits for value (softmax applied in loss function)

## AlphaZero Bridge

**File**: `bridge.py`

The `AlphaZeroBridge` class adapts HRM/TRM models to work with AlphaZero training:

```python
from model.bridge import AlphaZeroBridge

# Wrap any HRM/TRM model
bridge = AlphaZeroBridge(model)

# Forward pass
boards = torch.randn(batch_size, 112, 8, 8)  # Chess board encoding
pi, v, moves_left, q_info = bridge(boards)
```

**Responsibilities**:
- Converts AlphaZero board format to model format
- Manages carry state initialization and device placement
- Handles ACT looping until all sequences halt
- Extracts policy, value, and Q-learning outputs

## Output Heads

### Policy Heads

1. **Direct Policy** (`CastedLinear`): Simple linear projection
2. **Attention Policy** (`heads/attention_policy.py`): Attention-based policy over board squares

### Value Heads

1. **Direct Value** (`CastedLinear`): Single-token value prediction
2. **TensorFlow-Style Value** (`heads/value_heads.py`): Processes all spatial information

### Moves Left Head

Predicts remaining moves in the game (useful for time management).

## Common Utilities

### Layers (`common/layers.py`)

- `CastedLinear`: Linear layer with automatic dtype casting
- `CastedEmbedding`: Embedding layer with dtype casting
- `Attention`: Flash Attention implementation with RoPE support
- `RotaryEmbedding`: Rotary Position Embeddings (RoPE)
- `SwiGLU`: Gated Linear Unit with SiLU activation
- `LinearSwish`: Linear layer with Swish activation
- `rms_norm`: RMS Layer Normalization

### Initialization (`common/initialization.py`)

- `trunc_normal_init_`: JAX-style truncated normal initialization

### Sparse Embeddings (`common/sparse_embedding.py`)

- `CastedSparseEmbedding`: Memory-efficient sparse embeddings
- `CastedSparseEmbeddingSignSGD_Distributed`: Distributed optimizer for sparse embeddings

## Configuration Guide

### Common Configuration Parameters

```python
config = {
    # Model architecture
    'seq_len': 64,                    # Number of board squares
    'hidden_size': 512,               # Hidden dimension
    'num_heads': 8,                   # Attention heads
    'expansion': 2.0,                 # FFN expansion factor

    # HRM/TRM specific
    'H_cycles': 2,                    # High-level reasoning cycles
    'L_cycles': 3,                    # Low-level reasoning cycles
    'H_layers': 4,                    # High-level transformer layers
    'L_layers': 4,                    # Low-level transformer layers

    # ACT (Adaptive Computation Time)
    'halt_max_steps': 5,              # Maximum reasoning steps
    'halt_exploration_prob': 0.1,     # Exploration probability

    # Chess-specific
    'num_actions': 1858,              # Number of possible moves
    'square_feature_dim': 112,        # Features per square
    'board_x': 8,                     # Board width
    'board_y': 8,                     # Board height

    # Heads
    'use_attention_policy': True,     # Use attention-based policy
    'use_tensorflow_style_heads': True,  # Use TF-style value/moves heads

    # Positional encodings
    'pos_encodings': 'rope',          # 'rope' or 'learned'
    'rope_theta': 10000.0,            # RoPE base frequency

    # Precision
    'forward_dtype': 'bfloat16',      # Forward pass dtype
}
```

## Performance Considerations

### Memory Optimization

1. **bfloat16 Training**: Use `forward_dtype='bfloat16'` for memory efficiency
2. **Sparse Embeddings**: For large board representations
3. **Flash Attention**: Automatic memory-efficient attention

### Numerical Stability

1. **RMS Norm**: More stable than LayerNorm for bfloat16
2. **Residual Scaling**: Removed aggressive scaling to prevent overflow
3. **Truncated Normal Init**: Prevents extreme initial weights

## Citation

If you use this codebase, please cite:

```bibtex
@misc{hrm-chess,
  title={Hierarchical Reasoning Models for Chess},
  author={[Your Names]},
  year={2025},
  howpublished={\\url{https://github.com/yourusername/your-repo}}
}
```

## License

[Your License Here]

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Changelog

### Version 2.0.0 (2025-12-15)
- Complete refactoring of model directory
- Consolidated common code into `common/`
- Improved naming conventions
- Removed duplicate files
- Added comprehensive documentation
- Proper Python package structure with `__init__.py` files

### Version 1.0.0
- Initial implementation
