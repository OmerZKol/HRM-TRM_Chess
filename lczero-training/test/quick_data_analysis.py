#!/usr/bin/env python3

import sys
import os
import glob
import numpy as np
import torch

# Add paths
sys.path.append('/home/omerz/projects/ChessTypeBeat/lczero-training')
sys.path.append('/home/omerz/projects/ChessTypeBeat/lczero-training/tf')

from pytorch_train import ChessDataset
from policy_index import policy_index

def quick_analysis():
    """Quick analysis of training data patterns"""
    print("QUICK TRAINING DATA ANALYSIS")
    print("=" * 60)
    
    # Use the dataset that exists
    data_path = "/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run1--20250209-1017"
    chunk_files = glob.glob(os.path.join(data_path, "*.gz"))
    
    if not chunk_files:
        print("No chunk files found!")
        return
    
    print(f"Found {len(chunk_files)} chunk files")
    
    # Use small subset for quick analysis
    test_files = sorted(chunk_files)[:5]
    dataset = ChessDataset(test_files, sample_rate=100)  # Every 100th position
    
    print(f"Analyzing {len(dataset)} positions...")
    
    if len(dataset) == 0:
        print("No positions in dataset!")
        return
    
    # Analyze first 20 positions
    num_samples = min(20, len(dataset))
    
    print("\nSample Analysis:")
    print("-" * 40)
    
    legal_moves_counts = []
    policy_maxes = []
    win_probs = []
    draw_probs = []
    loss_probs = []
    moves_left_vals = []
    
    for i in range(num_samples):
        try:
            planes, policy, value, best_q, moves_left = dataset[i]
            
            # Policy analysis
            legal_moves = (policy > 0).sum().item()
            max_policy = policy.max().item()
            legal_moves_counts.append(legal_moves)
            policy_maxes.append(max_policy)
            
            # Value analysis
            win_probs.append(value[0].item())
            draw_probs.append(value[1].item())
            loss_probs.append(value[2].item())
            
            # Moves left
            moves_left_vals.append(moves_left.item())
            
            # Print sample details
            print(f"Position {i+1:2d}: {legal_moves:2d} legal moves, "
                  f"max policy {max_policy:.3f}, "
                  f"WDL [{value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}], "
                  f"moves left {moves_left.item():.0f}")
            
        except Exception as e:
            print(f"Error processing position {i}: {e}")
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS")
    print("-" * 40)
    print(f"Legal moves per position:")
    print(f"  Average: {np.mean(legal_moves_counts):.1f}")
    print(f"  Range: {min(legal_moves_counts)} - {max(legal_moves_counts)}")
    
    print(f"Policy confidence:")
    print(f"  Average max probability: {np.mean(policy_maxes):.3f}")
    print(f"  Range: {min(policy_maxes):.3f} - {max(policy_maxes):.3f}")
    
    print(f"Game outcomes (WDL):")
    print(f"  Win:  {np.mean(win_probs):.3f} ± {np.std(win_probs):.3f}")
    print(f"  Draw: {np.mean(draw_probs):.3f} ± {np.std(draw_probs):.3f}")
    print(f"  Loss: {np.mean(loss_probs):.3f} ± {np.std(loss_probs):.3f}")
    
    print(f"Game length:")
    print(f"  Average moves left: {np.mean(moves_left_vals):.1f}")
    print(f"  Range: {min(moves_left_vals):.0f} - {max(moves_left_vals):.0f}")

def show_top_moves():
    """Show most common moves in policy index"""
    print(f"\nCOMMON CHESS MOVES")
    print("-" * 40)
    
    # Common opening moves
    opening_moves = [
        "e2e4", "d2d4", "g1f3", "b1c3", "c2c4", "f1b5",  # Classical openings
        "e7e5", "d7d5", "g8f6", "b8c6", "c7c5",          # Black responses
        "o-o", "o-o-o"  # Castling (if encoded this way)
    ]
    
    print("Common opening moves in policy index:")
    for move in opening_moves:
        try:
            if move in policy_index:
                idx = policy_index.index(move)
                print(f"  {move:6s}: Index {idx:4d}")
            else:
                # Try some variations
                found = False
                for i, indexed_move in enumerate(policy_index):
                    if indexed_move.replace('-', '') == move.replace('-', ''):
                        print(f"  {move:6s}: Index {i:4d} (as {indexed_move})")
                        found = True
                        break
                if not found:
                    print(f"  {move:6s}: Not found")
        except:
            print(f"  {move:6s}: Error checking")
    
    # Show some random policy indices
    print(f"\nSample policy indices:")
    for i in [0, 50, 100, 200, 500, 1000, 1500, 1857]:
        if i < len(policy_index):
            print(f"  Index {i:4d}: {policy_index[i]}")

def analyze_one_position():
    """Detailed analysis of one position"""
    print(f"\nDETAILED POSITION ANALYSIS")
    print("-" * 40)
    
    data_path = "/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run1--20250209-1017"
    chunk_files = glob.glob(os.path.join(data_path, "*.gz"))[:1]
    
    if not chunk_files:
        return
    
    dataset = ChessDataset(chunk_files, sample_rate=1)
    
    if len(dataset) == 0:
        return
    
    planes, policy, value, best_q, moves_left = dataset[0]
    
    print(f"Position details:")
    print(f"  Board tensor shape: {planes.shape}")
    print(f"  Policy tensor shape: {policy.shape}")
    print(f"  Value (WDL): {value.numpy()}")
    print(f"  Moves left: {moves_left.item():.1f}")
    
    # Find legal moves
    legal_indices = torch.nonzero(policy > 0).flatten()
    legal_probs = policy[legal_indices]
    
    print(f"\nLegal moves: {len(legal_indices)}")
    
    # Sort by probability
    sorted_indices = torch.argsort(legal_probs, descending=True)
    
    print("Top 10 moves:")
    for i in range(min(10, len(sorted_indices))):
        move_idx = legal_indices[sorted_indices[i]].item()
        prob = legal_probs[sorted_indices[i]].item()
        move_name = policy_index[move_idx] if move_idx < len(policy_index) else f"Index {move_idx}"
        print(f"  {i+1:2d}. {move_name:8s}: {prob:.6f}")
    
    # Analyze board planes
    print(f"\nBoard representation:")
    active_planes = 0
    for i in range(planes.shape[0]):
        if planes[i].sum() > 0:
            active_planes += 1
    print(f"  Active planes: {active_planes}/112")
    print(f"  Total active squares: {(planes > 0).sum().item()}")

def main():
    print("QUICK CHESS TRAINING DATA ANALYSIS")
    print("=" * 80)
    
    try:
        quick_analysis()
        show_top_moves()
        analyze_one_position()
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()