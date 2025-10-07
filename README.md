# HRM-Chess

Hierarchical Reasoning Model (HRM) adapted for chess. Trains models on Leela Chess Zero (LC0) format training data to predict moves, board evaluation, and moves left in the game.

## Quick Start

### Setup
- Clone the repository
- Navigate to the `lczero-training` directory
```bash
cd lczero-training
```
- This project requires Python 3.12.3
### Requirements
install dependencies:
```bash
pip install -r requirements.txt
```
### Dataset
Download the LC0 training data
```bash
cd lczero-training
mkdir data
cd data
wget https://storage.lczero.org/files/training_data/training-run1--20250209-1017.tar
tar -xf training-run1--20250209-1017.tar
```

### Training with LC0 Data Format

```bash
cd lczero-training

# Train Simple CNN model
python pytorch_train.py --config config/simple_chess_nn.yaml --data-path data/training-run1--20250209-1017

# Train Transformer model
python pytorch_train.py --config config/transformer_chess_nn.yaml --data-path data/training-run1--20250209-1017

# Train HRM model (single-step, halt_max_steps=1)
python pytorch_train.py --config config/hrm_halt1.yaml --data-path data/training-run1--20250209-1017

# Train HRM model (multi-step adaptive, halt_max_steps=10)
python pytorch_train.py --config config/hrm_halt10.yaml --data-path data/training-run1--20250209-1017

# View training logs with TensorBoard
tensorboard --logdir runs/
```

## Model Architectures

### 1. Simple Chess Net (`simple_chess_nn.py`)
- **Architecture**: Basic CNN with 3 convolutional layers
- **Config**: [config/simple_chess_nn.yaml](lczero-training/config/simple_chess_nn.yaml)
- **Use case**: Baseline model, fast training

### 2. Transformer Chess Net (`transformer_chess_nn.py`)
- **Architecture**: Transformer with self-attention across all 64 board squares
- **Config**: [config/transformer_chess_nn.yaml](lczero-training/config/transformer_chess_nn.yaml)
- **Features**:
  - RoPE positional embeddings
  - Attention-based policy head
  - 512 hidden dim, 6 layers, 8 heads

### 3. HRM Model (`model/ChessNNet.py`)
- **Architecture**: Hierarchical Reasoning Model with Adaptive Computation Time (ACT)
- **Configs**:
  - [config/hrm_halt1.yaml](lczero-training/config/hrm_halt1.yaml) - Single-step (deterministic)
  - [config/hrm_halt10.yaml](lczero-training/config/hrm_halt10.yaml) - Multi-step adaptive reasoning
- **Features**:
  - Two-level hierarchical reasoning (H-level and L-level)
  - Adaptive computation with Q-learning based halting
  - TensorFlow-style spatial processing heads

## Data Format

The training uses **Leela Chess Zero (LC0) format** training data:
- **Input**: 112-channel 8×8 board representation
  - 104 planes: Piece positions (6 piece types × 2 colors × 8 history positions)
  - 8 planes: Auxiliary information (castling rights, en passant, rule50, etc.)
- **Outputs**:
  - **Policy**: 1858 possible moves (queen moves from each square)
  - **Value**: Win-Draw-Loss (WDL) probabilities
  - **Moves left**: Estimated plies remaining in the game

Data files are in `.gz` compressed binary format with LC0 V6 training record structure.

## Key Files

### Training Infrastructure
- [lczero-training/pytorch_train.py](lczero-training/pytorch_train.py) - Main training script with data loading
- [lczero-training/simple_chess_nn.py](lczero-training/simple_chess_nn.py) - Simple CNN baseline + loss functions
- [lczero-training/transformer_chess_nn.py](lczero-training/transformer_chess_nn.py) - Transformer architecture
- [lczero-training/model/ChessNNet.py](lczero-training/model/ChessNNet.py) - HRM model wrapper
- [lczero-training/model/HRMBridge.py](lczero-training/model/HRMBridge.py) - Bridge between LC0 format and HRM

### HRM Core Implementation
- [lczero-training/model/HRM_model/models/hrm/hrm_act_v1.py](lczero-training/model/HRM_model/models/hrm/hrm_act_v1.py) - HRM with ACT implementation
- [lczero-training/model/attention_policy_map.py](lczero-training/model/attention_policy_map.py) - Attention-based policy head
- [lczero-training/model/tensorflow_style_heads.py](lczero-training/model/tensorflow_style_heads.py) - Value/moves left heads

### Configuration
- [lczero-training/config/](lczero-training/config/) - Model-specific YAML configs

## Configuration Parameters

### HRM-Specific Parameters
- `halt_max_steps`: Maximum reasoning steps (1 = single-step, >1 = adaptive)
- `halt_exploration_prob`: Probability of random exploration during training
- `H_cycles`, `L_cycles`: Number of hierarchical reasoning cycles
- `H_layers`, `L_layers`: Depth of each hierarchical level
- `use_attention_policy`: Use attention mechanism for policy (vs. direct logits)
- `use_tensorflow_style_heads`: Use spatial processing for value/moves heads

## Legacy Training Scripts

The repository also contains legacy training code for move sequence prediction:
- `old_implementation/run_chess_training.py` - Original HRM training with move sequences
- `old_implementation/dataset/build_chess_dataset.py` - PGN to HRM format conversion
- `old_implementation/chess_puzzle_dataset.py` - Move sequence dataset loader

These are kept for reference but the recommended approach is using the LC0 format training above.
