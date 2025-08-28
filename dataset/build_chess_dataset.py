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
import chess

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
        # Move encoding: from_square (64) * to_square (64) + 64 * 4 (promotion) 
        # REPRESENTING PROMOTION SEPARATE TO REGULAR MOVES
        self.max_moves = 64 * 64 + 64 * 4  # 4352 possible moves
        # REGULAR_MOVES = list(range(0, 4096)) # 64*64
        # QUEEN_PROMOTIONS = list(range(4096, 4096 + 64))
        # ROOK_PROMOTIONS = list(range(4096 + 64, 4096 + 128))  
        # BISHOP_PROMOTIONS = list(range(4096 + 128, 4096 + 192))
        # KNIGHT_PROMOTIONS = list(range(4096 + 192, 4096 + 256))
        self.promotion_pieces = ["q", "r", "b", "n"]
    
    def encode_promotion_move(self, from_square, to_square, promotion_piece):
        # Verify it's a valid promotion square
        if not (to_square < 8 or to_square >= 56):
            raise ValueError(f"Invalid promotion square: {to_square}")
        
        # each 'from square' has 3 'potential' 'to square' squares
        # Direct mapping for the 16 promotion squares
        if to_square < 8:  # Rank 1
            base_offset = to_square
        else:  # Rank 8
            base_offset = to_square - 56 + 8
        
        if from_square < 16: # rank 2
            from_offset = from_square - 8
        else:   # rank 7
            from_offset = from_square - 48 + 8

        relative_to_offset = from_offset - base_offset + 1
        # base_offset is now in the range [0, 15]
        if promotion_piece == chess.QUEEN:
            return 4096 + from_offset * 3 + relative_to_offset # 4096-4144
        if promotion_piece == chess.ROOK:
            from_offset += 16*3
            return 4096 + from_offset * 3 + relative_to_offset # 4144-4192
        elif promotion_piece == chess.BISHOP:
            from_offset += 16*3
            return 4096 + from_offset * 3 + relative_to_offset # 4192-4240
        elif promotion_piece == chess.KNIGHT:
            from_offset += 16*3
            return 4096 + from_offset * 3 + relative_to_offset # 4240-4288

    def encode_move(self, board, move):
        """Convert a chess move object to a move index for the model."""
        #convert to from - to move indexes
        from_square = move.from_square
        to_square = move.to_square

        #Handlge promotion
        if (move.promotion):
            promotion_piece = move.promotion
            return self.encode_promotion_move(from_square, to_square, promotion_piece)
        else:
            return from_square * 64 + to_square
        
    def decode_move(self, move_id: int) -> str:
        """Decode integer back to move string (UCI format)."""
        if move_id < 0 or move_id >= self.max_moves:
            raise ValueError(f"Invalid move ID: {move_id}")
        
        # Regular moves (0-4095): from_square * 64 + to_square
        if move_id < 4096:
            from_square = move_id // 64
            to_square = move_id % 64
            
            from_str = chess.square_name(from_square)
            to_str = chess.square_name(to_square)

            return f"{from_str}{to_str}"

        return self.decode_promotion(move_id)

    def decode_promotion(self, move_id):
        # Promotion moves (4096-4288)
        promotion_offset = move_id - 4096
        
        # Determine piece type based on offset ranges
        if promotion_offset < 16*3:  # 0-47: Queen promotions
            base_offset = promotion_offset
            piece_type = 0
        elif promotion_offset < 16*6:  # 48-95: Rook promotions  
            base_offset = promotion_offset - 16*3
            piece_type = 1
        elif promotion_offset < 16*9:  # 96-143: Bishop promotions
            base_offset = promotion_offset - 16*6
            piece_type = 2
        elif promotion_offset < 16*12:  # 144-191: Knight promotions
            base_offset = promotion_offset - 16*9
            piece_type = 3
        else:
            raise ValueError(f"Invalid promotion move ID: {move_id}")
        
        # Extract from_offset and relative_to_offset from base_offset
        from_offset = base_offset // 3
        relative_to_offset = base_offset % 3
                
        # Convert from_offset back to actual from_square
        if from_offset < 8:  # Rank 2
            from_square = from_offset + 8  # a2-h2 gives 8-16 range
            # Calculate to_square on rank 1
            base_to_offset = from_offset  # Corresponding square on rank 1
            to_square = base_to_offset + (relative_to_offset - 1)  # Adjust for relative offset
            # Ensure to_square is valid (0-7 for rank 1)
            if to_square < 0:
                to_square = 0
            elif to_square > 7:
                to_square = 7
        else:  # Rank 7
            from_square = (from_offset - 8) + 48  # a7-h7 gives 48-56 range
            # Calculate to_square on rank 8
            base_to_offset = (from_offset - 8) + 56  # Corresponding square on rank 8
            to_square = base_to_offset + (relative_to_offset - 1)  # Adjust for relative offset
            # Ensure to_square is valid (56-63 for rank 8)
            if to_square < 56:
                to_square = 56
            elif to_square > 63:
                to_square = 63
        
        from_str = chess.square_name(from_square)
        to_str = chess.square_name(to_square)
        
        # Map piece type to promotion character
        promotion_chars = ['q', 'r', 'b', 'n']
        promotion_char = promotion_chars[piece_type]
        
        return f"{from_str}{to_str}{promotion_char}"


class SimplePGNParser:
    """Simple PGN parser"""
    
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
            if token:
                moves.append(token)
        return moves

    def convert_moves_and_boards(self, moves):
        """Convert a sequence of moves to a list of board states and 
        also returns a list of move objects"""
        board = chess.Board()
        boards = [board.copy()]
        converted_moves = []
        for move in moves:
            move_object = board.parse_san(move)  # Convert SAN to move object
            converted_moves.append(move_object)
            board.push(move_object)
            boards.append(board.copy())
        return boards, converted_moves

def create_move_mask(possible_moves, num_actions):
        """Create binary mask for valid moves.
        
        Args:
            possible_moves: List of lists, each containing valid move indices for a position
            num_actions: Total number of possible actions in the vocabulary
            
        Returns:
            np array: Binary mask of shape (batch_size, num_actions) where 1 = valid move, 0 = invalid
        """
        mask = np.zeros((num_actions), dtype=np.float32)
        mask[possible_moves] = True
        return mask

def calculate_position_value(result_str: str, move_index: int, total_moves: int, side_to_move: bool) -> float:
    """Calculate estimated position value based on game result.
    
    Args:
        result_str: Game result ("1-0", "0-1", "1/2-1/2")
        move_index: Current move number (0-based)
        total_moves: Total number of moves in the game
        side_to_move: True if white to move, False if black to move
        
    Returns:
        Float value between -1 and 1 representing position evaluation
        (1 = winning for white, -1 = winning for black, 0 = equal)
    """
    # Parse game result
    if result_str == "1-0":  # White wins
        final_value = 1.0
    elif result_str == "0-1":  # Black wins
        final_value = -1.0
    elif result_str == "1/2-1/2":  # Draw
        final_value = 0.0
    else:
        final_value = 0.0  # Default for unknown results
    
    # Calculate how far through the game we are (0.0 to 1.0)
    game_progress = move_index / max(total_moves - 1, 1)
    
    # Early in the game, positions are closer to equal (0)
    # Later in the game, they approach the final result
    # Use a sigmoid-like curve to gradually approach the final value
    
    # Interpolate between neutral (0.0) and final result
    # Early positions have less certain outcomes
    uncertainty_factor = 1.0 - (0.8 * (1.0 - game_progress))  # More uncertainty early
    estimated_value = final_value * uncertainty_factor
    
    # Add some noise based on game phase to account for tactical complexity
    # Opening/endgame tend to be more predictable than middlegame
    if game_progress < 0.2 or game_progress > 0.8:
        # Opening or endgame - more predictable
        noise_factor = 0.9
    else:
        # Middlegame - more tactical uncertainty
        noise_factor = 0.7
    
    estimated_value *= noise_factor
    
    # From black's perspective, flip the value
    if not side_to_move:
        estimated_value = -estimated_value
    
    # Ensure value stays in [-1, 1] range
    return max(-1.0, min(1.0, estimated_value))

def create_position_encoding(board_history: List[chess.Board], seq_len: int = 64) -> List[List[int]]:
    """Create square-centric position encoding from chess board history.
    
    Args:
        board_history: List of board states (current + up to 7 previous positions)
        seq_len: Target sequence length (64 squares)
    
    Returns:
        List of 64 subarrays, each containing all information about that specific square
        Each square's array contains: [historical_pieces, castling_info, ep_info, other_features]
    """
    
    # Ensure we have at most 8 positions (current + 7 previous)
    if len(board_history) > 8:
        board_history = board_history[-8:]
    
    # Get current board (most recent)
    current_board = board_history[-1] if board_history else chess.Board()
    
    # Initialize encoding: 64 squares, each containing all information about that square
    square_encodings = []
    
    # Process each of the 64 squares
    for square in chess.SQUARES:
        square_info = []
        
        # ===== HISTORICAL PIECE INFORMATION (96 features per square) =====
        # 8 time steps × 12 piece types = 96 features
        
        for time_step in range(8):
            # Get board for this time step (or empty board if not available)
            if time_step < len(board_history):
                board = board_history[-(time_step + 1)]  # Most recent first
            else:
                board = chess.Board()
                board.clear()  # Empty board for missing history
            
            # Check what piece (if any) is on this square for each piece type
            # White pieces (6 features): Pawns, Rooks, Knights, Bishops, Queens, King
            for piece_type in [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]:
                piece = board.piece_at(square)
                if piece and piece.piece_type == piece_type and piece.color == chess.WHITE:
                    square_info.append(1)
                else:
                    square_info.append(0)
            
            # Black pieces (6 features): Pawns, Rooks, Knights, Bishops, Queens, King  
            for piece_type in [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]:
                piece = board.piece_at(square)
                if piece and piece.piece_type == piece_type and piece.color == chess.BLACK:
                    square_info.append(1)
                else:
                    square_info.append(0)
        
        # ===== GAME STATE INFORMATION (16 features per square) =====
        # These are the same for all squares but included for each square
        
        # Castling Rights (4 features)
        square_info.append(1 if current_board.has_kingside_castling_rights(chess.WHITE) else 0)
        square_info.append(1 if current_board.has_queenside_castling_rights(chess.WHITE) else 0)
        square_info.append(1 if current_board.has_kingside_castling_rights(chess.BLACK) else 0)
        square_info.append(1 if current_board.has_queenside_castling_rights(chess.BLACK) else 0)
        
        # En Passant Information (1 feature)
        ep_square = current_board.ep_square
        square_info.append(1 if ep_square is not None and square == ep_square else 0)
        
        # Repetition Information (8 features) - simplified
        # For now, set all to 0 (would need full game history to detect repetitions)
        for _ in range(8):
            square_info.append(0)
        
        # Rule50 Counter (1 feature)
        rule50_value = min(current_board.halfmove_clock / 100.0, 1.0)  # Normalize to [0,1]
        square_info.append(int(rule50_value * 255))  # Scale to 0-255 range
        
        # Side to Move (1 feature)
        square_info.append(1 if current_board.turn == chess.WHITE else 0)
        
        # All-ones Constant (1 feature)
        square_info.append(1)
        
        # Add this square's information to the encoding
        square_encodings.append(square_info)
    
    # Verify we have exactly 64 squares
    if len(square_encodings) != 64:
        print(f"Warning: square count {len(square_encodings)} != expected 64")
        # Pad or truncate to 64 squares
        while len(square_encodings) < 64:
            square_encodings.append([0] * 112)  # Add empty square with 112 features
        square_encodings = square_encodings[:64]
    
    # Verify each square has exactly 112 features (96 historical + 16 game state)
    expected_features = 96 + 16  # 112 features per square
    for i, square_info in enumerate(square_encodings):
        if len(square_info) != expected_features:
            print(f"Warning: square {i} has {len(square_info)} features != expected {expected_features}")
            # Pad or truncate to correct number of features
            while len(square_info) < expected_features:
                square_info.append(0)
            square_encodings[i] = square_info[:expected_features]
    
    return square_encodings


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
        max_elo_range = 3600 # Arbitrary upper limit for now
        step = (max_elo_range - min_elo_range) // num_groups
        for i in range(num_groups):
            start = min_elo_range + i * step
            end = min_elo_range + (i + 1) * step if i < num_groups - 1 else max_elo_range
            elo_ranges.append((start, end))
    
    print(f"Creating {num_groups} groups with ELO ranges: {elo_ranges}")
    
    # Track data by group
    train_data_by_group = {i: {"inputs": [], "targets": [], "move_targets": [], "puzzle_ids": [], "possible_moves": [], "value_targets": []} for i in range(num_groups)}
    test_data_by_group = {i: {"inputs": [], "targets": [], "move_targets": [], "puzzle_ids": [], "possible_moves": [], "value_targets": []} for i in range(num_groups)}

    # Square-centric input: 64 squares, each with 112 features
    position_history_len = 64  # 64 squares as sequence length
    vocab_size = encoder.max_moves
    
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

            if len(row) < 3:
                continue

            pgn_str, elo_str, result_str = row[0], row[1], row[2]
            
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

            # Get board states and the converted moves
            boards, converted_moves = parser.convert_moves_and_boards(moves)
            
            # Create training examples: predict next move from position
            for move_idx in range(min(len(converted_moves) - 1, max_moves_per_game)):
                # get corresponding board state for that move
                board = boards[move_idx]
                # get possible moves from this board state
                possible_moves = list(board.legal_moves)

                # Create board history (current + up to 7 previous positions)
                board_history = []
                for hist_idx in range(max(0, move_idx - 7), move_idx + 1):
                    if hist_idx < len(boards):
                        board_history.append(boards[hist_idx])
                
                # Reverse so most recent is last (expected by create_position_encoding)
                board_history.reverse()

                # Input: LC0-style board state encoding
                position_encoding = create_position_encoding(board_history, position_history_len)

                # Target: next move
                move = converted_moves[move_idx]
                #encoded move
                move_target = encoder.encode_move(board, move)

                #Encoded possible moves (going to be used for masking)
                encoded_possible_moves = [encoder.encode_move(board, m) for m in possible_moves]
                mask = create_move_mask(encoded_possible_moves, encoder.max_moves)

                # Calculate position value based on game result
                total_moves_in_game = len(converted_moves)
                side_to_move = board.turn  # True for white, False for black
                value_target = calculate_position_value(result_str, move_idx, total_moves_in_game, side_to_move)

                # HRM format
                inputs = position_encoding
                labels = [-100] * len(inputs)  # No sequence generation labels needed
                puzzle_id = games_per_group[group_id]  # Unique puzzle ID within group

                # Split train/test
                is_train = (valid_positions % 10) < (train_ratio * 10)  # Simple split
                
                if is_train:
                    train_data_by_group[group_id]["inputs"].append(inputs)
                    train_data_by_group[group_id]["targets"].append(labels)
                    train_data_by_group[group_id]["move_targets"].append(move_target)
                    train_data_by_group[group_id]["possible_moves"].append(mask)
                    train_data_by_group[group_id]["puzzle_ids"].append(puzzle_id)
                    train_data_by_group[group_id]["value_targets"].append(value_target)
                else:
                    test_data_by_group[group_id]["inputs"].append(inputs)
                    test_data_by_group[group_id]["targets"].append(labels)
                    test_data_by_group[group_id]["move_targets"].append(move_target)
                    test_data_by_group[group_id]["possible_moves"].append(mask)
                    test_data_by_group[group_id]["puzzle_ids"].append(puzzle_id)
                    test_data_by_group[group_id]["value_targets"].append(value_target)
                
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
        all_possible_moves = []
        all_value_targets = []
        
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
                    # if current_puzzle != -1:
                    #     current_puzzle_id += 1
                    current_puzzle = puzzle_id
                    puzzle_indices.append(len(all_inputs))
                
                all_inputs.append(group_data["inputs"][idx])
                all_labels.append(group_data["targets"][idx])
                all_move_targets.append(group_data["move_targets"][idx])
                all_puzzle_identifiers.append(0)
                all_possible_moves.append(group_data["possible_moves"][idx])
                all_value_targets.append(group_data["value_targets"][idx])
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
        np.save(os.path.join(save_dir, "all__possible_moves.npy"), np.array(all_possible_moves, dtype=np.int32))  # Object array for variable-length lists
        np.save(os.path.join(save_dir, "all__value_targets.npy"), np.array(all_value_targets, dtype=np.float32))

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
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
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
            "ignore_label_id": 0,
            "blank_identifier_id": 0,
            "vocab_size": vocab_size,
            "seq_len": position_history_len,
            "num_puzzle_identifiers": 1,
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
    parser.add_argument("--max-games", type=int, default=1000,
                       help="Maximum number of games to process")
    parser.add_argument("--min-elo", type=int, default=2700,
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