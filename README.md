# HRM-Chess

Hierarchical Reasoning Model (HRM) and Tiny Recursive Model (TRM) adapted for chess. Trains models on Leela Chess Zero (LC0) format training data to predict moves, board evaluation, and moves left in the game. Overall, neither the inclusion of the hierarchical reasoning nor the recursive refinement structure yielded improvements over a transformer baseline in the experiments.

## Training Results

![Validation metrics for all models](src/plots/overview_validation_all_models.png)

![Validation metrics for TRM and Transformer models](src/plots/overview_validation_trm_+_transformer_models.png)

Training results are available in the `src/runs/` directory.
```bash
pip install tensorboard
tensorboard --logdir src/runs/
```

## Quick Start

```bash
pip install -r requirements.txt
```
Change version according to the available GPU:
```bash
pip install flash_attn==2.8.3 --no-build-isolation
```

### Dataset
Download LC0 training data:
```bash
cd src/data
wget https://storage.lczero.org/files/training_data/training-run1--20250209-1017.tar
tar -xf training-run1--20250209-1017.tar
```

### Training

```bash
cd src
# Train a single model
python pytorch_train.py --config config/hrm/hrm_halt1_single_cycle.yaml --data-path data/training-run1--20250209-1017

# Train all configs sequentially
python batch_train.py --config-dir config/ --data-path data/training-run1--20250209-1017

# View training logs
tensorboard --logdir runs/
```

## Model Architectures

| Model | Description | Config |
|-------|-------------|--------|
| **Simple CNN** | 3-layer CNN baseline | `config/cnn/simple_chess_nn.yaml` |
| **Transformer** | Self-attention across 64 squares with RoPE | `config/transformer/transformer_attn_p_head.yaml` |
| **HRM** | Hierarchical reasoning with H-level and L-level | `config/hrm/hrm_halt1_single_cycle.yaml`, `hrm_halt1_2cycle.yaml` |
| **TRM** | Simplified HRM model | `config/trm/trm_halt1_single_cycle.yaml`, `trm_halt1_3cycle.yaml`, `trm_halt3.yaml` |

### Key Features
- **Adaptive Computation Time (ACT)**: Models can halt after N reasoning steps
- **Hierarchical Reasoning**: Two-level attention (H-level and L-level)
- **Recursive Refinement**: HRM/TRM models refine outputs over multiple internal cycles
- **Attention-based Policy Head**: Maps attention weights to 1858 legal chess moves

## Data Format

Uses **Leela Chess Zero (LC0)** binary format:
- **Input**: 112-channel 8x8 board (piece positions + history + auxiliary info)
- **Outputs**: Policy (1858 moves), Value (WDL), Moves left

## Project Structure

```
src/
├── pytorch_train.py      # Main training script
├── batch_train.py        # Sequential multi-config training
├── chess_dataset.py      # LC0 data parser
├── chess_loss.py         # Loss functions
├── config/               # YAML configs (cnn/, hrm/, trm/, transformer/)
├── model/
│   ├── hrm/hrm_model.py  # HRM implementation
│   ├── trm/trm_model.py  # TRM implementation
│   ├── heads/            # Policy and value heads
│   └── common/           # Shared layers (RoPE, attention, etc.)
└── data/                 # Training datasets
```

## Configuration

Key parameters in YAML configs:
- `halt_max_steps`: Max reasoning steps (1 = single-step, >1 = adaptive)
- `H_cycles`, `L_cycles`: Number of hierarchical reasoning cycles
- `hidden_size`, `num_heads`: Model dimensions
