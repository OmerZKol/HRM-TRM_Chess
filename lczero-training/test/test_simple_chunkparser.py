#!/usr/bin/env python3

import sys
import os
import glob
import numpy as np
import struct
import gzip

# Add tf directory to path
sys.path.append('/home/omerz/projects/ChessTypeBeat/lczero-training/tf')

def test_direct_parsing():
    """Test parsing using the direct methods from chunkparser.py"""
    print("TESTING DIRECT CHUNK PARSING")
    print("=" * 80)
    
    # Import the parsing functions directly
    from chunkparser import ChunkParser
    
    # Get a single chunk file
    data_path = "/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run1--20250209-1017"
    chunk_files = glob.glob(os.path.join(data_path, "*.gz"))
    
    if not chunk_files:
        print("No chunk files found!")
        return
    
    test_file = chunk_files[0]
    print(f"Testing with file: {os.path.basename(test_file)}")
    
    # Create a minimal ChunkParser instance for access to conversion methods
    try:
        parser = ChunkParser.__new__(ChunkParser)  # Create without calling __init__
        parser.expected_input_format = 1  # Set manually
        parser.init_structs()  # Initialize the struct parsers
        
        # Set up flat planes
        parser.flat_planes = {
            0: (64 * 4) * b'\x00',  # All zeros plane
            1: (64 * 4) * b'\x01'   # All ones plane  
        }
        
        print("ChunkParser methods accessible!")
        
        # Read and parse the chunk file directly
        with gzip.open(test_file, 'rb') as f:
            chunk_data = f.read()
            
        print(f"Chunk data size: {len(chunk_data)} bytes")
        
        # Get first few records
        version = chunk_data[0:4]
        version_int = struct.unpack('i', version)[0]
        print(f"Version: {version_int}")
        
        record_size = parser.v6_struct.size
        print(f"Record size: {record_size} bytes")
        
        num_records = min(3, len(chunk_data) // record_size)
        print(f"Testing first {num_records} records...")
        
        for i in range(num_records):
            record = chunk_data[i * record_size:(i + 1) * record_size]
            
            print(f"\n--- RECORD {i + 1} ---")
            
            try:
                # Use the conversion method directly
                planes, probs, winner, best_q, plies_left = parser.convert_v6_to_tuple(record)
                
                print(f"Conversion successful!")
                print(f"Planes length: {len(planes)} bytes")
                print(f"Policy length: {len(probs)} bytes")
                print(f"Winner: {winner}")
                print(f"Best Q: {best_q}")
                print(f"Plies left: {plies_left}")
                
                # Parse the binary data for analysis
                planes_array = np.frombuffer(planes, dtype=np.float32).reshape(112, 8, 8)
                policy_array = np.frombuffer(probs, dtype=np.float32)
                winner_array = np.frombuffer(winner, dtype=np.float32)
                best_q_array = np.frombuffer(best_q, dtype=np.float32)
                plies_left_val = np.frombuffer(plies_left, dtype=np.float32)[0]
                
                print(f"Analysis:")
                print(f"  Planes shape: {planes_array.shape}")
                print(f"  Planes active: {np.count_nonzero(planes_array)}")
                print(f"  Policy shape: {policy_array.shape}")
                print(f"  Policy sum: {policy_array.sum():.6f}")
                print(f"  Policy max: {policy_array.max():.6f}")
                print(f"  Policy non-zero: {np.count_nonzero(policy_array)}")
                print(f"  Winner (WDL): [{winner_array[0]:.3f}, {winner_array[1]:.3f}, {winner_array[2]:.3f}]")
                print(f"  Best Q (WDL): [{best_q_array[0]:.3f}, {best_q_array[1]:.3f}, {best_q_array[2]:.3f}]")
                print(f"  Plies left: {plies_left_val:.1f}")
                
                # Show top moves
                top_indices = np.argsort(policy_array)[-5:][::-1]
                print(f"  Top 5 moves:")
                for j, idx in enumerate(top_indices):
                    print(f"    {j+1}. Index {idx}: {policy_array[idx]:.6f}")
                
            except Exception as e:
                print(f"Error converting record {i+1}: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"Error setting up parser: {e}")
        import traceback
        traceback.print_exc()

def compare_with_pytorch():
    """Compare TF parsing with PyTorch implementation"""
    print("\n" + "=" * 80)
    print("COMPARING TF vs PyTorch PARSING")
    print("=" * 80)
    
    data_path = "/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run1--20250209-1017"
    chunk_files = glob.glob(os.path.join(data_path, "*.gz"))[:1]
    
    if not chunk_files:
        print("No chunk files found!")
        return
    
    print("PyTorch implementation:")
    print("-" * 40)
    
    try:
        sys.path.append('/home/omerz/projects/ChessTypeBeat/lczero-training')
        from chess_dataset import ChessDataset
        
        dataset = ChessDataset(chunk_files, sample_rate=1)
        if len(dataset) > 0:
            planes_pt, policy_pt, winner_pt, best_q_pt, moves_left_pt = dataset[0]
            
            print(f"Planes shape: {planes_pt.shape}")
            print(f"Policy shape: {policy_pt.shape}")
            print(f"Policy sum: {policy_pt.sum():.6f}")
            print(f"Policy max: {policy_pt.max():.6f}")
            print(f"Policy non-zero: {(policy_pt != 0).sum().item()}")
            print(f"Winner: {winner_pt.numpy()}")
            print(f"Moves left: {moves_left_pt.numpy()}")
            
            # Show top moves
            top_values, top_indices = policy_pt.topk(5)
            print(f"Top 5 moves:")
            for j, (idx, val) in enumerate(zip(top_indices, top_values)):
                print(f"  {j+1}. Index {idx}: {val:.6f}")
        else:
            print("No data in PyTorch dataset")
            
    except Exception as e:
        print(f"PyTorch error: {e}")
        import traceback
        traceback.print_exc()

def analyze_policy_encoding():
    """Analyze how policy probabilities are encoded"""
    print("\n" + "=" * 80)
    print("ANALYZING POLICY ENCODING")
    print("=" * 80)
    
    # Load policy index mapping
    try:
        from policy_index import policy_index
        print(f"Policy index loaded: {len(policy_index)} moves")
        
        # Show some example moves
        print("Example move indices:")
        for i in range(0, min(20, len(policy_index)), 4):
            print(f"  Index {i:3d}: {policy_index[i]}")
            
        # Common opening moves
        common_moves = ["e2e4", "d2d4", "g1f3", "b1c3", "f1c4"]
        print(f"\nCommon opening move indices:")
        for move in common_moves:
            if move in policy_index:
                idx = policy_index.index(move)
                print(f"  {move}: Index {idx}")
            else:
                print(f"  {move}: Not found")
                
    except ImportError:
        print("Could not import policy_index - using analysis")
        
        # Analyze policy distribution from actual data
        data_path = "/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run1--20250209-1017"
        chunk_files = glob.glob(os.path.join(data_path, "*.gz"))[:3]
        
        if chunk_files:
            try:
                sys.path.append('/home/omerz/projects/ChessTypeBeat/lczero-training')
                from chess_dataset import ChessDataset
                
                dataset = ChessDataset(chunk_files, sample_rate=10)  # Sample every 10th
                
                # Analyze multiple positions
                all_nonzero = []
                all_maxvals = []
                
                for i in range(min(20, len(dataset))):
                    _, policy, _, _, _ = dataset[i]
                    nonzero_count = (policy != 0).sum().item()
                    max_val = policy.max().item()
                    all_nonzero.append(nonzero_count)
                    all_maxvals.append(max_val)
                
                print(f"Policy statistics across {len(all_nonzero)} positions:")
                print(f"  Average legal moves: {np.mean(all_nonzero):.1f}")
                print(f"  Legal move range: {min(all_nonzero)} - {max(all_nonzero)}")
                print(f"  Average max probability: {np.mean(all_maxvals):.4f}")
                print(f"  Max probability range: {min(all_maxvals):.4f} - {max(all_maxvals):.4f}")
                
            except Exception as e:
                print(f"Error analyzing policy: {e}")

def main():
    print("COMPREHENSIVE CHUNK PARSING TEST")
    print("=" * 80)
    
    # 1. Test direct parsing methods
    test_direct_parsing()
    
    # 2. Compare with PyTorch
    compare_with_pytorch()
    
    # 3. Analyze policy encoding
    analyze_policy_encoding()

if __name__ == "__main__":
    main()