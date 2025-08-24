# HRM-Chess

Hierarchical Reasoning Model (HRM) adapted for chess move prediction. Currently uses Adaptive Computation Time (ACT) to learn next move prediction from move sequences.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build chess dataset from PGN games
python dataset/build_chess_dataset.py

# Train the model
python run_chess_training.py

# Visualize dataset (optional)
python run_dataset_visualization.py
```

## Current Model Architecture

- **Input**: Last 100 moves (adjustable) in a chess game (encoded as integers)
- **Output**: Next best move prediction
- **Method**: Hierarchical reasoning with variable computation time
- **Dataset**: High-ELO chess games (2200+ rating (adjustable)) converted to move sequences

## Key Features

- Move sequence modeling (not board state)
- Adaptive computation for complex positions
- Hierarchical reasoning (strategic + tactical levels)
- Next move prediction task

## Key Files
### Core Training
- `run_chess_training.py` - Main training script with HRM configuration
- `chess_puzzle_dataset.py` - Chess dataset loader with move targets
- `config/cfg_chess.yaml` - Training hyperparameters and model config

### Dataset
- `dataset/build_chess_dataset.py` - Converts PGN games stored in `data/chess_games.csv` to HRM format
- `run_dataset_visualization.py` - Visualise dataset statistics and samples

### Utilities
- `test_chess_dataset.py` - Dataset instantiation tests
