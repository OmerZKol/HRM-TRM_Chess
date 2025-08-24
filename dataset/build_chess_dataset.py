"""
Build chess dataset for HRM move prediction training.
Converts PGN games to input/target pairs for next move prediction.
"""

import os
import re
import csv
import json
import argparse
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Simple chess move encoding without external dependencies
SQUARES = [f"{file}{rank}" for rank in "12345678" for file in "abcdefgh"]
SQUARE_TO_IDX = {square: i for i, square in enumerate(SQUARES)}
IDX_TO_SQUARE = {i: square for i, square in enumerate(SQUARES)}

# Piece symbols
PIECES = "PNBRQK"
PIECE_TO_IDX = {piece: i for i, piece in enumerate(PIECES)}

@dataclass
class ChessPosition:
    """Simplified chess position representation."""
    board: str  # FEN-like board representation
    to_move: str  # 'w' or 'b'
    castling: str  # Castling rights
    ep_square: Optional[str]  # En passant square
    halfmove: int
    fullmove: int


class ChessMoveEncoder:
    """Encodes chess moves as integers for HRM training."""
    
    def __init__(self):
        # Move encoding: from_square (64) * to_square (64) * promotion_piece (5: none, Q, R, B, N)
        self.max_moves = 64 * 64 * 5  # 20,480 possible moves
        self.promotion_pieces = ["", "q", "r", "b", "n"]
    
    def encode_move(self, move_str: str) -> int:
        """Encode a move string (e.g., 'e2e4', 'e7e8q') to integer."""
        move_str = move_str.lower().strip()
        
        # Handle castling
        if move_str in ["o-o", "o-o-o", "0-0", "0-0-0"]:
            # Simplified: treat as king moves
            if move_str in ["o-o", "0-0"]:
                return self.encode_move("e1g1")  # White short castling
            else:
                return self.encode_move("e1c1")  # White long castling
        
        # Extract from/to squares
        if len(move_str) < 4:
            return 0  # Invalid move
        
        from_square = move_str[:2]
        to_square = move_str[2:4]
        promotion = move_str[4:5] if len(move_str) > 4 else ""
        
        if from_square not in SQUARE_TO_IDX or to_square not in SQUARE_TO_IDX:
            return 0  # Invalid squares
        
        from_idx = SQUARE_TO_IDX[from_square]
        to_idx = SQUARE_TO_IDX[to_square]
        promo_idx = self.promotion_pieces.index(promotion) if promotion in self.promotion_pieces else 0
        
        # Encode as: from * 64 * 5 + to * 5 + promotion
        move_id = from_idx * 64 * 5 + to_idx * 5 + promo_idx
        return min(move_id, self.max_moves - 1)
    
    def decode_move(self, move_id: int) -> str:
        """Decode integer back to move string."""
        if move_id >= self.max_moves:
            return "invalid"
        
        promo_idx = move_id % 5
        move_id //= 5
        to_idx = move_id % 64
        from_idx = move_id // 64
        
        from_square = IDX_TO_SQUARE.get(from_idx, "a1")
        to_square = IDX_TO_SQUARE.get(to_idx, "a1")
        promotion = self.promotion_pieces[promo_idx]
        
        return from_square + to_square + promotion


class SimplePGNParser:
    """Simple PGN parser without external chess library."""
    
    def __init__(self):
        self.move_encoder = ChessMoveEncoder()
    
    def parse_game(self, pgn_str: str) -> List[str]:
        """Parse PGN string and extract moves."""
        # Remove comments, variations, and annotations
        pgn_clean = re.sub(r'\{[^}]*\}', '', pgn_str)  # Remove comments
        pgn_clean = re.sub(r'\([^)]*\)', '', pgn_clean)  # Remove variations
        pgn_clean = re.sub(r'[!?+#]+', '', pgn_clean)  # Remove annotations
        
        # Extract moves (remove move numbers and result)
        moves = []
        tokens = pgn_clean.split()
        
        for token in tokens:
            # Skip move numbers, results, and empty tokens
            if not token or token.endswith('.') or token in ['1-0', '0-1', '1/2-1/2', '*']:
                continue
            
            # Convert algebraic notation to UCI-like format (simplified)
            move = self._algebraic_to_uci(token)
            if move:
                moves.append(move)
        
        return moves
    
    def _algebraic_to_uci(self, algebraic: str) -> Optional[str]:
        """Convert algebraic notation to UCI-like format (very simplified)."""
        algebraic = algebraic.strip()
        
        # Handle castling
        if algebraic in ['O-O', '0-0']:
            return 'e1g1'  # Simplified: assume white castling
        if algebraic in ['O-O-O', '0-0-0']:
            return 'e1c1'  # Simplified: assume white long castling
        
        # For simplicity, this is a very basic conversion
        # In production, you'd need a full chess engine
        # For now, we'll extract square names when possible
        squares = re.findall(r'[a-h][1-8]', algebraic)
        if len(squares) >= 2:
            return squares[-2] + squares[-1]  # Last two squares found
        elif len(squares) == 1:
            # Pawn move or ambiguous - simplified handling
            return squares[0] + squares[0]  # Placeholder
        
        return None


def create_position_encoding(moves: List[str], max_history: int = 10) -> List[int]:
    """Create a simple position encoding from move history."""
    encoder = ChessMoveEncoder()
    
    # Encode recent moves (simplified position representation)
    encoded_moves = []
    for move in moves[-max_history:]:
        encoded_moves.append(encoder.encode_move(move))
    
    # Pad or truncate to fixed length
    while len(encoded_moves) < max_history:
        encoded_moves.insert(0, 0)  # Pad with empty moves
    
    return encoded_moves[:max_history]


def build_chess_dataset(
    csv_path: str,
    output_dir: str,
    max_games: Optional[int] = None,
    min_elo: int = 2000,
    max_moves_per_game: int = 60,
    train_ratio: float = 0.8,
    num_groups: int = 10
):
    """Build HRM-compatible chess dataset from CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create splits
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    parser = SimplePGNParser()
    encoder = ChessMoveEncoder()
    
    # Group games by ELO ranges for curriculum learning
    elo_ranges = []
    if num_groups == 3:
        elo_ranges = [(2000, 2200), (2200, 2400), (2400, 3000)]  # Easy, Medium, Hard
    elif num_groups == 4:
        elo_ranges = [(2000, 2150), (2150, 2300), (2300, 2450), (2450, 3000)]
    else:
        # Create equal ELO ranges
        min_elo_range = min_elo
        max_elo_range = 4000 # Arbitrary upper limit for now
        step = (max_elo_range - min_elo_range) // num_groups
        for i in range(num_groups):
            start = min_elo_range + i * step
            end = min_elo_range + (i + 1) * step if i < num_groups - 1 else max_elo_range
            elo_ranges.append((start, end))
    
    print(f"Creating {num_groups} groups with ELO ranges: {elo_ranges}")
    
    # Track data by group
    train_data_by_group = {i: {"inputs": [], "targets": [], "move_targets": [], "puzzle_ids": []} for i in range(num_groups)}
    test_data_by_group = {i: {"inputs": [], "targets": [], "move_targets": [], "puzzle_ids": []} for i in range(num_groups)}
    
    position_history_len = 10  # Number of recent moves to include
    vocab_size = encoder.max_moves + 100  # Buffer for special tokens
    
    print(f"Processing chess games from {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        game_count = 0
        valid_positions = 0
        games_per_group = {i: 0 for i in range(num_groups)}
        
        for row in tqdm(reader, desc="Processing games"):
            if max_games and game_count >= max_games:
                break
            
            if len(row) < 2:
                continue
                
            pgn_str, elo_str = row[0], row[1]
            
            # Filter by ELO and assign to group
            try:
                elo = int(elo_str)
                if elo < min_elo:
                    continue
            except:
                continue
            
            # Determine group based on ELO
            group_id = None
            for i, (min_range, max_range) in enumerate(elo_ranges):
                if min_range <= elo < max_range:
                    group_id = i
                    break
            
            if group_id is None:
                continue  # ELO doesn't fit in any group
            
            # Parse moves
            moves = parser.parse_game(pgn_str)
            if len(moves) < 10:  # Skip very short games
                continue
            
            # Create training examples: predict next move from position
            for move_idx in range(min(len(moves) - 1, max_moves_per_game)):
                # Input: recent move history
                position_encoding = create_position_encoding(moves[:move_idx + 1], position_history_len)
                
                # Target: next move
                next_move = moves[move_idx + 1]
                move_target = encoder.encode_move(next_move)
                
                # HRM format
                inputs = position_encoding
                labels = [-100] * len(inputs)  # No sequence generation labels needed
                puzzle_id = games_per_group[group_id] % 1000  # Unique puzzle ID within group
                
                # Split train/test
                is_train = (valid_positions % 10) < (train_ratio * 10)  # Simple split
                
                if is_train:
                    train_data_by_group[group_id]["inputs"].append(inputs)
                    train_data_by_group[group_id]["targets"].append(labels)
                    train_data_by_group[group_id]["move_targets"].append(move_target)
                    train_data_by_group[group_id]["puzzle_ids"].append(puzzle_id)
                else:
                    test_data_by_group[group_id]["inputs"].append(inputs)
                    test_data_by_group[group_id]["targets"].append(labels)
                    test_data_by_group[group_id]["move_targets"].append(move_target)
                    test_data_by_group[group_id]["puzzle_ids"].append(puzzle_id)
                
                valid_positions += 1
            
            games_per_group[group_id] += 1
            
            game_count += 1
    
    # Print statistics by group
    total_train = sum(len(train_data_by_group[i]["inputs"]) for i in range(num_groups))
    total_test = sum(len(test_data_by_group[i]["inputs"]) for i in range(num_groups))
    
    print(f"Processed {game_count} games, created {valid_positions} training positions")
    print(f"Train: {total_train}, Test: {total_test}")
    
    for i in range(num_groups):
        train_count = len(train_data_by_group[i]["inputs"])
        test_count = len(test_data_by_group[i]["inputs"])
        print(f"  Group {i} (ELO {elo_ranges[i][0]}-{elo_ranges[i][1]}): Train={train_count}, Test={test_count}")

    def save_multi_group_data(data_by_group, save_dir, num_groups):
        """Save multi-group data in HRM format."""
        if not any(len(data_by_group[i]["inputs"]) > 0 for i in range(num_groups)):
            return

        # Combine all groups into single arrays, but maintain group boundaries
        all_inputs = []
        all_labels = []
        all_move_targets = []
        all_puzzle_identifiers = []
        
        # Create group_indices: [start_group0, start_group1, ..., end]
        group_indices = [0]
        puzzle_indices = [0]  # Maps puzzle_id -> start_index in flat arrays
        
        current_puzzle_id = 0
        
        for group_id in range(num_groups):
            group_data = data_by_group[group_id]
            if len(group_data["inputs"]) == 0:
                continue
                
            # Sort by puzzle_id within group to create proper puzzle boundaries
            group_size = len(group_data["inputs"])
            indices = list(range(group_size))
            indices.sort(key=lambda i: group_data["puzzle_ids"][i])
            
            # Add data in puzzle_id order
            current_puzzle = -1
            for idx in indices:
                puzzle_id = group_data["puzzle_ids"][idx]
                
                # New puzzle started
                if puzzle_id != current_puzzle:
                    if current_puzzle != -1:
                        current_puzzle_id += 1
                    current_puzzle = puzzle_id
                    puzzle_indices.append(len(all_inputs))
                
                all_inputs.append(group_data["inputs"][idx])
                all_labels.append(group_data["targets"][idx])
                all_move_targets.append(group_data["move_targets"][idx])
                all_puzzle_identifiers.append(current_puzzle_id)
            
            # Update group boundary
            group_indices.append(current_puzzle_id + 1)
        
        # Final boundary for puzzle_indices
        puzzle_indices.append(len(all_inputs))
        
        # Save arrays
        np.save(os.path.join(save_dir, "all__inputs.npy"), np.array(all_inputs, dtype=np.int32))
        np.save(os.path.join(save_dir, "all__labels.npy"), np.array(all_labels, dtype=np.int32))
        np.save(os.path.join(save_dir, "all__move_targets.npy"), np.array(all_move_targets, dtype=np.int32))
        np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), np.array(all_puzzle_identifiers, dtype=np.int32))
        np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), np.array(puzzle_indices, dtype=np.int32))
        np.save(os.path.join(save_dir, "all__group_indices.npy"), np.array(group_indices, dtype=np.int32))
    
    # Save training and test data
    save_multi_group_data(train_data_by_group, train_dir, num_groups)
    save_multi_group_data(test_data_by_group, test_dir, num_groups)
    
    # Save metadata in HRM-compatible format
    dataset_info = {
        # Basic info
        "vocab_size": vocab_size,
        "seq_len": position_history_len,
        "num_actions": encoder.max_moves,
        "num_games": game_count,
        "num_positions": valid_positions,
        "train_size": total_train,
        "test_size": total_test,
        
        # HRM-required metadata fields
        "pad_id": 0,
        "ignore_label_id": -100,
        "blank_identifier_id": 999,
        "num_puzzle_identifiers": 1000,
        "total_groups": num_groups,  # Multiple groups for curriculum learning
        "mean_puzzle_examples": valid_positions / max(game_count, 1),
        "sets": ["all"],  # Single dataset set
        
        # Group information
        "elo_ranges": elo_ranges,
        "groups": {str(i): {"elo_range": elo_ranges[i], "description": f"ELO {elo_ranges[i][0]}-{elo_ranges[i][1]}"} for i in range(num_groups)}
    }
    
    with open(os.path.join(output_dir, "dataset.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Save train/test dataset.json with HRM-compatible metadata
    for split_dir, size in [(train_dir, total_train), (test_dir, total_test)]:
        split_info = {
            # HRM-required fields
            "pad_id": 0,
            "ignore_label_id": -100,
            "blank_identifier_id": 999,
            "vocab_size": vocab_size,
            "seq_len": position_history_len,
            "num_puzzle_identifiers": 1000,
            "total_groups": num_groups,
            "mean_puzzle_examples": size / max(game_count, 1),
            "sets": ["all"],
            
            # Additional info
            "size": size,
            "num_actions": encoder.max_moves,
            "elo_ranges": elo_ranges,
            "groups": {str(i): {"elo_range": elo_ranges[i], "description": f"ELO {elo_ranges[i][0]}-{elo_ranges[i][1]}"} for i in range(num_groups)}
        }
        
        with open(os.path.join(split_dir, "dataset.json"), 'w') as f:
            json.dump(split_info, f, indent=2)
    
    print(f"Dataset saved to {output_dir}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Sequence length: {position_history_len}")
    print(f"Number of possible moves: {encoder.max_moves}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build chess dataset for HRM")
    parser.add_argument("--csv-path", default="data/chess_games_more_filtered.csv", 
                       help="Path to chess games CSV")
    parser.add_argument("--output-dir", default="data/chess-move-prediction", 
                       help="Output directory")
    parser.add_argument("--max-games", type=int, default=100000,
                       help="Maximum number of games to process")
    parser.add_argument("--min-elo", type=int, default=2200,
                       help="Minimum ELO rating")
    parser.add_argument("--max-moves-per-game", type=int, default=150,
                       help="Maximum moves per game to use")
    parser.add_argument("--num-groups", type=int, default=10,
                       help="Number of difficulty groups to create")
    
    args = parser.parse_args()
    
    build_chess_dataset(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        max_games=args.max_games,
        min_elo=args.min_elo,
        max_moves_per_game=args.max_moves_per_game,
        num_groups=args.num_groups
    )