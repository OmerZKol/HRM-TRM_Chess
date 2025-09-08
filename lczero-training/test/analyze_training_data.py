#!/usr/bin/env python3

import sys
import os
import glob
import numpy as np
import torch
from collections import defaultdict, Counter

# Add paths
sys.path.append('/home/omerz/projects/ChessTypeBeat/lczero-training')
sys.path.append('/home/omerz/projects/ChessTypeBeat/lczero-training/tf')

from pytorch_train import ChessDataset
from policy_index import policy_index

def analyze_policy_patterns(dataset, num_samples=100):
    """Analyze policy distribution patterns"""
    print("POLICY PATTERN ANALYSIS")
    print("=" * 60)
    
    move_frequencies = Counter()
    position_legal_moves = []
    policy_entropies = []
    max_probabilities = []
    
    for i in range(min(num_samples, len(dataset))):
        if i % 20 == 0:
            print(f"Processing sample {i+1}/{min(num_samples, len(dataset))}")
            
        try:
            _, policy, _, _, _ = dataset[i]
            
            # Find legal moves (non-negative probabilities)
            legal_mask = policy > 0
            legal_moves = policy[legal_mask]
            legal_indices = torch.nonzero(legal_mask).flatten()
            
            # Statistics
            position_legal_moves.append(len(legal_moves))
            
            if len(legal_moves) > 0:
                # Calculate entropy
                probs = legal_moves / legal_moves.sum()
                entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
                policy_entropies.append(entropy)
                max_probabilities.append(legal_moves.max().item())
                
                # Track move frequencies
                for idx in legal_indices:
                    move_frequencies[idx.item()] += 1
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"\nPolicy Statistics:")
    print(f"  Average legal moves per position: {np.mean(position_legal_moves):.1f}")
    print(f"  Legal moves range: {min(position_legal_moves)} - {max(position_legal_moves)}")
    print(f"  Average policy entropy: {np.mean(policy_entropies):.3f}")
    print(f"  Average max probability: {np.mean(max_probabilities):.4f}")
    
    # Most common moves
    print(f"\nMost common legal moves:")
    for move_idx, frequency in move_frequencies.most_common(10):
        move_name = policy_index[move_idx] if move_idx < len(policy_index) else f"Index {move_idx}"
        percentage = (frequency / num_samples) * 100
        print(f"  {move_name:8s}: {frequency:4d} positions ({percentage:5.1f}%)")
    
    return move_frequencies

def analyze_value_patterns(dataset, num_samples=100):
    """Analyze value (WDL) distribution patterns"""
    print(f"\nVALUE PATTERN ANALYSIS")
    print("=" * 60)
    
    win_probs = []
    draw_probs = []
    loss_probs = []
    
    result_counts = {"win": 0, "draw": 0, "loss": 0}
    
    for i in range(min(num_samples, len(dataset))):
        try:
            _, _, value, _, _ = dataset[i]
            
            win_probs.append(value[0].item())
            draw_probs.append(value[1].item())
            loss_probs.append(value[2].item())
            
            # Classify result
            max_idx = torch.argmax(value).item()
            if max_idx == 0:
                result_counts["win"] += 1
            elif max_idx == 1:
                result_counts["draw"] += 1
            else:
                result_counts["loss"] += 1
                
        except Exception as e:
            print(f"Error processing value {i}: {e}")
            continue
    
    print(f"Value (WDL) Statistics:")
    print(f"  Average Win probability:  {np.mean(win_probs):.4f} ± {np.std(win_probs):.4f}")
    print(f"  Average Draw probability: {np.mean(draw_probs):.4f} ± {np.std(draw_probs):.4f}")
    print(f"  Average Loss probability: {np.mean(loss_probs):.4f} ± {np.std(loss_probs):.4f}")
    
    print(f"\nGame Results:")
    total = sum(result_counts.values())
    for result, count in result_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {result.capitalize():5s}: {count:4d} positions ({percentage:5.1f}%)")

def analyze_board_patterns(dataset, num_samples=50):
    """Analyze board representation patterns"""
    print(f"\nBOARD PATTERN ANALYSIS")
    print("=" * 60)
    
    plane_activations = np.zeros(112)
    piece_counts = []
    
    for i in range(min(num_samples, len(dataset))):
        try:
            planes, _, _, _, _ = dataset[i]
            
            # Count active planes
            for plane_idx in range(112):
                if planes[plane_idx].sum() > 0:
                    plane_activations[plane_idx] += 1
            
            # Count total pieces (approximate)
            total_pieces = (planes > 0).sum().item()
            piece_counts.append(total_pieces)
            
        except Exception as e:
            print(f"Error processing board {i}: {e}")
            continue
    
    print(f"Board Statistics:")
    print(f"  Average active features: {np.mean(piece_counts):.1f}")
    print(f"  Active feature range: {min(piece_counts)} - {max(piece_counts)}")
    
    print(f"\nMost active planes (likely piece types):")
    active_planes = [(i, count) for i, count in enumerate(plane_activations) if count > 0]
    active_planes.sort(key=lambda x: x[1], reverse=True)
    
    for plane_idx, count in active_planes[:20]:
        percentage = (count / num_samples) * 100
        print(f"  Plane {plane_idx:3d}: {count}/{num_samples} positions ({percentage:5.1f}%)")

def analyze_moves_left_patterns(dataset, num_samples=100):
    """Analyze moves left predictions"""
    print(f"\nMOVES LEFT ANALYSIS")
    print("=" * 60)
    
    moves_left_values = []
    
    for i in range(min(num_samples, len(dataset))):
        try:
            _, _, _, _, moves_left = dataset[i]
            moves_left_values.append(moves_left.item())
        except Exception as e:
            print(f"Error processing moves left {i}: {e}")
            continue
    
    if moves_left_values:
        print(f"Moves Left Statistics:")
        print(f"  Average: {np.mean(moves_left_values):.1f} plies")
        print(f"  Range: {min(moves_left_values):.1f} - {max(moves_left_values):.1f} plies")
        print(f"  Std Dev: {np.std(moves_left_values):.1f} plies")
        
        # Histogram
        hist, bins = np.histogram(moves_left_values, bins=10)
        print(f"\nDistribution:")
        for i in range(len(hist)):
            percentage = (hist[i] / len(moves_left_values)) * 100
            print(f"  {bins[i]:6.1f}-{bins[i+1]:6.1f}: {hist[i]:3d} ({percentage:4.1f}%)")

def find_interesting_positions(dataset, num_samples=50):
    """Find positions with interesting characteristics"""
    print(f"\nINTERESTING POSITIONS")
    print("=" * 60)
    
    # Find positions with very few legal moves (tactical)
    # Find positions with very even policy distribution (complex)
    # Find positions with extreme evaluations
    
    interesting = {
        "few_moves": [],      # < 10 legal moves
        "many_moves": [],     # > 50 legal moves  
        "decisive": [],       # Win/Loss > 0.8
        "balanced": [],       # All WDL values close to 0.33
        "confident": []       # Max policy > 0.7
    }
    
    for i in range(min(num_samples, len(dataset))):
        try:
            planes, policy, value, _, moves_left = dataset[i]
            
            legal_moves = (policy > 0).sum().item()
            max_policy = policy.max().item()
            max_value = value.max().item()
            
            if legal_moves < 10:
                interesting["few_moves"].append((i, legal_moves, max_policy))
            if legal_moves > 50:
                interesting["many_moves"].append((i, legal_moves, max_policy))
            if max_value > 0.8:
                interesting["decisive"].append((i, value.numpy(), max_policy))
            if abs(value[0] - 0.33) < 0.1 and abs(value[1] - 0.33) < 0.1:
                interesting["balanced"].append((i, value.numpy(), legal_moves))
            if max_policy > 0.7:
                interesting["confident"].append((i, max_policy, legal_moves))
                
        except Exception as e:
            continue
    
    for category, positions in interesting.items():
        if positions:
            print(f"\n{category.replace('_', ' ').title()} positions:")
            for pos_data in positions[:3]:  # Show top 3
                if category == "few_moves":
                    idx, legal, max_pol = pos_data
                    print(f"  Position {idx}: {legal} legal moves, max policy {max_pol:.4f}")
                elif category == "decisive":
                    idx, values, max_pol = pos_data
                    print(f"  Position {idx}: WDL [{values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f}], max policy {max_pol:.4f}")
                elif category == "confident":
                    idx, max_pol, legal = pos_data
                    print(f"  Position {idx}: max policy {max_pol:.4f}, {legal} legal moves")
                else:
                    print(f"  Position {pos_data[0]}: {pos_data[1:]}")

def main():
    # Use the newer dataset path
    data_path = "/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run1--20250209-1017"
    chunk_files = glob.glob(os.path.join(data_path, "*.gz"))
    
    if not chunk_files:
        print("No chunk files found!")
        return
    
    print("COMPREHENSIVE TRAINING DATA ANALYSIS")
    print("=" * 80)
    print(f"Data path: {data_path}")
    print(f"Total files: {len(chunk_files)}")
    
    # Use a subset for analysis
    test_files = sorted(chunk_files)[:10]  # First 10 files
    print(f"Analyzing first {len(test_files)} files...")
    
    try:
        # Load dataset with light sampling
        dataset = ChessDataset(test_files, sample_rate=50)  # Every 50th position
        print(f"Dataset loaded: {len(dataset)} positions")
        
        if len(dataset) == 0:
            print("No data in dataset!")
            return
        
        # Run analyses
        move_frequencies = analyze_policy_patterns(dataset, num_samples=200)
        analyze_value_patterns(dataset, num_samples=200)
        analyze_board_patterns(dataset, num_samples=100)
        analyze_moves_left_patterns(dataset, num_samples=200)
        find_interesting_positions(dataset, num_samples=100)
        
        print(f"\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()