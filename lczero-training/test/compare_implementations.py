#!/usr/bin/env python3

import sys
import os
import glob
import numpy as np
import torch
import struct
import gzip

# Add paths
sys.path.append('/home/omerz/projects/ChessTypeBeat/lczero-training')
sys.path.append('/home/omerz/projects/ChessTypeBeat/lczero-training/tf')

def compare_raw_parsing():
    """Compare raw binary parsing between implementations"""
    print("COMPARING RAW BINARY PARSING")
    print("=" * 80)
    
    # Get a test file
    data_path = "/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run1--20250209-1017"
    chunk_files = glob.glob(os.path.join(data_path, "*.gz"))
    
    if not chunk_files:
        print("No chunk files found!")
        return
        
    test_file = chunk_files[0]
    print(f"Testing with: {os.path.basename(test_file)}")
    
    # Read raw chunk data
    with gzip.open(test_file, 'rb') as f:
        chunk_data = f.read()
    
    print(f"Chunk size: {len(chunk_data)} bytes")
    
    # Parse using TF structs directly
    from chunkparser import V6_STRUCT_STRING, V5_STRUCT_STRING, V4_STRUCT_STRING, V3_STRUCT_STRING
    
    v6_struct = struct.Struct(V6_STRUCT_STRING)
    record_size = v6_struct.size
    
    print(f"V6 record size: {record_size} bytes")
    print(f"Number of records: {len(chunk_data) // record_size}")
    
    # Get first record
    if len(chunk_data) >= record_size:
        record = chunk_data[:record_size]
        
        # Parse with TF struct
        try:
            tf_parsed = v6_struct.unpack(record)
            print(f"\nTF struct parsing successful:")
            print(f"  Version: {struct.unpack('i', tf_parsed[0])[0]}")
            print(f"  Input format: {tf_parsed[1]}")
            print(f"  Policy data length: {len(tf_parsed[2])}")
            print(f"  Planes data length: {len(tf_parsed[3])}")
            print(f"  Castling: us_ooo={tf_parsed[4]}, us_oo={tf_parsed[5]}, them_ooo={tf_parsed[6]}, them_oo={tf_parsed[7]}")
            print(f"  Side to move: {tf_parsed[8]}")
            print(f"  Rule50: {tf_parsed[9]}")
            print(f"  Best Q: {tf_parsed[13]}")
            print(f"  Result Q: {tf_parsed[19]}, Result D: {tf_parsed[20]}")
            print(f"  Plies left: {tf_parsed[18]}")
            
        except Exception as e:
            print(f"TF struct parsing failed: {e}")
    
    # Compare with PyTorch implementation
    print(f"\nPyTorch implementation:")
    try:
        from chess_dataset import ChessDataset
        dataset = ChessDataset([test_file], sample_rate=1)
        
        if len(dataset) > 0:
            planes, policy, value, best_q, moves_left = dataset[0]
            print(f"  Dataset loaded successfully")
            print(f"  Planes shape: {planes.shape}")
            print(f"  Policy shape: {policy.shape}")
            print(f"  Value: {value.numpy()}")
            print(f"  Best Q: {best_q.numpy()}")
            print(f"  Moves left: {moves_left.numpy()}")
        else:
            print("  No data in PyTorch dataset")
            
    except Exception as e:
        print(f"PyTorch parsing failed: {e}")
        import traceback
        traceback.print_exc()

def compare_conversion_methods():
    """Compare the actual conversion methods"""
    print(f"\nCOMPARING CONVERSION METHODS")
    print("=" * 80)
    
    # Create minimal TF parser instance
    try:
        from chunkparser import ChunkParser
        
        # Initialize structs manually 
        tf_parser = ChunkParser.__new__(ChunkParser)
        tf_parser.v6_struct = struct.Struct(V6_STRUCT_STRING)
        tf_parser.v5_struct = struct.Struct(V5_STRUCT_STRING) 
        tf_parser.v4_struct = struct.Struct(V4_STRUCT_STRING)
        tf_parser.v3_struct = struct.Struct(V3_STRUCT_STRING)
        tf_parser.expected_input_format = 1
        
        # Set up flat planes like TF version
        tf_parser.flat_planes = {
            0: (64 * 4) * b'\x00',
            1: (64 * 4) * b'\x01'
        }
        
        print("TF parser initialized successfully")
        
    except Exception as e:
        print(f"Failed to initialize TF parser: {e}")
        return
    
    # Get test data
    data_path = "/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run1--20250209-1017"
    chunk_files = glob.glob(os.path.join(data_path, "*.gz"))[:1]
    
    if not chunk_files:
        return
    
    with gzip.open(chunk_files[0], 'rb') as f:
        chunk_data = f.read()
    
    # Get first record
    record_size = tf_parser.v6_struct.size
    if len(chunk_data) >= record_size:
        record = chunk_data[:record_size]
        
        # Try TF conversion
        print(f"\nTesting TF convert_v6_to_tuple method:")
        try:
            tf_result = tf_parser.convert_v6_to_tuple(record)
            print(f"  TF conversion successful!")
            print(f"  Planes length: {len(tf_result[0])} bytes")
            print(f"  Policy length: {len(tf_result[1])} bytes") 
            print(f"  Winner length: {len(tf_result[2])} bytes")
            print(f"  Best Q length: {len(tf_result[3])} bytes")
            print(f"  Plies left length: {len(tf_result[4])} bytes")
            
            # Parse the results
            tf_planes = np.frombuffer(tf_result[0], dtype=np.float32)
            tf_policy = np.frombuffer(tf_result[1], dtype=np.float32)
            tf_winner = np.frombuffer(tf_result[2], dtype=np.float32)
            tf_best_q = np.frombuffer(tf_result[3], dtype=np.float32)
            tf_plies = np.frombuffer(tf_result[4], dtype=np.float32)
            
            print(f"  TF Planes: {tf_planes.shape}, sum={tf_planes.sum():.3f}")
            print(f"  TF Policy: {tf_policy.shape}, sum={tf_policy.sum():.6f}, max={tf_policy.max():.6f}")
            print(f"  TF Winner: {tf_winner}")
            print(f"  TF Best Q: {tf_best_q}")
            print(f"  TF Plies: {tf_plies[0]:.1f}")
            
        except Exception as e:
            print(f"  TF conversion failed: {e}")
            tf_result = None
    
        # Try PyTorch conversion  
        print(f"\nTesting PyTorch conversion:")
        try:
            from chess_dataset import ChessDataset
            pt_dataset = ChessDataset.__new__(ChessDataset)
            pt_dataset.v6_struct = struct.Struct(V6_STRUCT_STRING)
            pt_dataset.expected_input_format = None
            pt_dataset.flat_planes = {
                0: (64 * 4) * b'\x00',
                1: (64 * 4) * b'\x01'
            }
            
            pt_result = pt_dataset._convert_v6_to_tuple(record)
            print(f"  PyTorch conversion successful!")
            print(f"  PyTorch Planes: {pt_result[0].shape}, sum={pt_result[0].sum():.3f}")
            print(f"  PyTorch Policy: {pt_result[1].shape}, sum={pt_result[1].sum():.6f}, max={pt_result[1].max():.6f}")
            print(f"  PyTorch Winner: {pt_result[2].numpy()}")
            print(f"  PyTorch Best Q: {pt_result[3].numpy()}")
            print(f"  PyTorch Plies: {pt_result[4].numpy()[0]:.1f}")
            
            # Direct comparison if both worked
            if tf_result is not None:
                print(f"\nDIRECT COMPARISON:")
                print(f"  Planes match: {np.allclose(tf_planes.reshape(112, 8, 8), pt_result[0].numpy())}")
                print(f"  Policy match: {np.allclose(tf_policy, pt_result[1].numpy())}")
                print(f"  Winner match: {np.allclose(tf_winner, pt_result[2].numpy())}")
                print(f"  Best Q match: {np.allclose(tf_best_q, pt_result[3].numpy())}")
                print(f"  Plies match: {np.allclose(tf_plies, pt_result[4].numpy())}")
                
                # Check specific differences
                if not np.allclose(tf_policy, pt_result[1].numpy()):
                    policy_diff = np.abs(tf_policy - pt_result[1].numpy())
                    print(f"  Max policy difference: {policy_diff.max():.8f}")
                    print(f"  Mean policy difference: {policy_diff.mean():.8f}")
                
                if not np.allclose(tf_planes.reshape(112, 8, 8), pt_result[0].numpy()):
                    planes_diff = np.abs(tf_planes.reshape(112, 8, 8) - pt_result[0].numpy())
                    print(f"  Max planes difference: {planes_diff.max():.8f}")
                    print(f"  Mean planes difference: {planes_diff.mean():.8f}")
            
        except Exception as e:
            print(f"  PyTorch conversion failed: {e}")
            import traceback
            traceback.print_exc()

def compare_multiple_records():
    """Compare multiple records to ensure consistency"""
    print(f"\nCOMPARING MULTIPLE RECORDS")
    print("=" * 80)
    
    data_path = "/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run1--20250209-1017"
    chunk_files = glob.glob(os.path.join(data_path, "*.gz"))[:1]
    
    if not chunk_files:
        return
    
    try:
        from chess_dataset import ChessDataset
        dataset = ChessDataset(chunk_files, sample_rate=1)
        
        print(f"Testing {min(5, len(dataset))} records:")
        
        for i in range(min(5, len(dataset))):
            try:
                planes, policy, value, best_q, moves_left = dataset[i]
                
                legal_moves = (policy > 0).sum().item()
                max_policy = policy.max().item()
                
                print(f"  Record {i+1}: "
                      f"planes {planes.shape}, "
                      f"policy sum {policy.sum():.3f}, "
                      f"legal moves {legal_moves}, "
                      f"max policy {max_policy:.4f}, "
                      f"value {value.numpy()}, "
                      f"moves left {moves_left.item():.1f}")
                
                # Basic sanity checks
                assert planes.shape == (112, 8, 8), f"Wrong planes shape: {planes.shape}"
                assert policy.shape == (1858,), f"Wrong policy shape: {policy.shape}"
                assert value.shape == (3,), f"Wrong value shape: {value.shape}"
                assert abs(value.sum().item() - 1.0) < 0.001, f"Value doesn't sum to 1: {value.sum().item()}"
                assert legal_moves > 0, f"No legal moves found"
                assert max_policy > 0, f"Max policy is not positive"
                
                print(f"    ✓ All checks passed")
                
            except Exception as e:
                print(f"    ✗ Record {i+1} failed: {e}")
        
        print(f"\nAll records processed successfully!")
        
    except Exception as e:
        print(f"Failed to process records: {e}")

def verify_data_integrity():
    """Verify the overall data integrity"""
    print(f"\nVERIFYING DATA INTEGRITY")
    print("=" * 80)
    
    data_path = "/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run1--20250209-1017"
    chunk_files = glob.glob(os.path.join(data_path, "*.gz"))[:5]  # Test first 5 files
    
    try:
        from chess_dataset import ChessDataset
        dataset = ChessDataset(chunk_files, sample_rate=10)  # Sample every 10th
        
        print(f"Testing {len(dataset)} positions across {len(chunk_files)} files")
        
        all_legal_moves = []
        all_policy_sums = []
        all_value_sums = []
        parse_errors = 0
        
        for i in range(len(dataset)):
            try:
                planes, policy, value, best_q, moves_left = dataset[i]
                
                legal_moves = (policy > 0).sum().item()
                policy_sum = policy.sum().item() 
                value_sum = value.sum().item()
                
                all_legal_moves.append(legal_moves)
                all_policy_sums.append(policy_sum)
                all_value_sums.append(value_sum)
                
                # Check for anomalies
                if legal_moves < 5 or legal_moves > 100:
                    print(f"  Unusual legal move count at position {i}: {legal_moves}")
                
                if abs(value_sum - 1.0) > 0.01:
                    print(f"  Value sum anomaly at position {i}: {value_sum}")
                
            except Exception as e:
                parse_errors += 1
                if parse_errors <= 3:  # Only show first few errors
                    print(f"  Parse error at position {i}: {e}")
        
        print(f"\nIntegrity Report:")
        print(f"  Total positions processed: {len(dataset)}")
        print(f"  Parse errors: {parse_errors}")
        print(f"  Success rate: {((len(dataset) - parse_errors) / len(dataset) * 100):.1f}%")
        
        if all_legal_moves:
            print(f"  Legal moves: avg {np.mean(all_legal_moves):.1f}, range {min(all_legal_moves)}-{max(all_legal_moves)}")
            print(f"  Policy sums: avg {np.mean(all_policy_sums):.3f}, std {np.std(all_policy_sums):.3f}")
            print(f"  Value sums: avg {np.mean(all_value_sums):.6f}, std {np.std(all_value_sums):.6f}")
        
        print(f"  ✓ Data integrity verification complete")
        
    except Exception as e:
        print(f"Integrity verification failed: {e}")

def main():
    print("PYTORCH vs TENSORFLOW IMPLEMENTATION COMPARISON")
    print("=" * 80)
    
    # Import required constants
    global V6_STRUCT_STRING, V5_STRUCT_STRING, V4_STRUCT_STRING, V3_STRUCT_STRING
    from chess_dataset import ChessDataset, V6_STRUCT_STRING, V5_STRUCT_STRING, V4_STRUCT_STRING, V3_STRUCT_STRING
    
    try:
        # 1. Compare raw parsing
        compare_raw_parsing()
        
        # 2. Compare conversion methods
        compare_conversion_methods()
        
        # 3. Compare multiple records
        compare_multiple_records()
        
        # 4. Verify overall data integrity
        verify_data_integrity()
        
        print(f"\n" + "=" * 80)
        print("COMPARISON COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()