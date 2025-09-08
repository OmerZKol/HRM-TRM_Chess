#!/usr/bin/env python3

import torch
import numpy as np
import struct
import gzip
import glob
import os
from pytorch_train import ChessDataset, V6_STRUCT_STRING

def inspect_single_record(record_bytes):
    """Inspect a single training record and show its contents"""
    v6_struct = struct.Struct(V6_STRUCT_STRING)
    
    try:
        # Unpack the V6 content
        (ver, input_format, probs, planes, us_ooo, us_oo, them_ooo, them_oo,
         stm, rule50_count, invariance_info, dep_result, root_q, best_q,
         root_d, best_d, root_m, best_m, plies_left, result_q, result_d,
         played_q, played_d, played_m, orig_q, orig_d, orig_m, visits,
         played_idx, best_idx, reserved1, reserved2, reserved3, reserved4) = v6_struct.unpack(record_bytes)
        
        print(f"Version: {struct.unpack('i', ver)[0]}")
        print(f"Input Format: {input_format}")
        print(f"Rule50 Count: {rule50_count}")
        print(f"Side to Move: {stm}")
        print(f"Castling - Us OOO: {us_ooo}, Us OO: {us_oo}")
        print(f"Castling - Them OOO: {them_ooo}, Them OO: {them_oo}")
        print(f"Invariance Info: {invariance_info}")
        print(f"Result (dep): {dep_result}")
        
        print(f"\nQ Values:")
        print(f"  Root Q: {root_q:.4f}, Best Q: {best_q:.4f}")
        print(f"  Root D: {root_d:.4f}, Best D: {best_d:.4f}")
        print(f"  Result Q: {result_q:.4f}, Result D: {result_d:.4f}")
        
        print(f"\nMoves:")
        print(f"  Root M: {root_m:.1f}, Best M: {best_m:.1f}")
        print(f"  Plies Left: {plies_left:.1f}")
        print(f"  Played Q: {played_q:.4f}, Played D: {played_d:.4f}, Played M: {played_m:.1f}")
        
        print(f"\nSearch Info:")
        print(f"  Visits: {visits}")
        print(f"  Played Index: {played_idx}, Best Index: {best_idx}")
        
        # Policy analysis
        probs_array = np.frombuffer(probs, dtype=np.float32)
        print(f"\nPolicy Distribution:")
        print(f"  Total moves: {len(probs_array)}")
        print(f"  Policy sum: {probs_array.sum():.6f}")
        print(f"  Max policy: {probs_array.max():.6f}")
        print(f"  Non-zero moves: {np.count_nonzero(probs_array)}")
        
        # Show top 5 moves
        top_indices = np.argsort(probs_array)[-5:][::-1]
        print(f"  Top 5 moves:")
        for i, idx in enumerate(top_indices):
            print(f"    {i+1}. Index {idx}: {probs_array[idx]:.6f}")
            
        # Plane analysis
        planes_array = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32)
        print(f"\nBoard Planes:")
        print(f"  Plane data length: {len(planes)} bytes ({len(planes_array)} bits)")
        print(f"  Active bits: {np.count_nonzero(planes_array)}")
        
        return True
        
    except Exception as e:
        print(f"Error unpacking record: {e}")
        return False

def inspect_chunk_file(filename, max_records=5):
    """Inspect a single chunk file"""
    print(f"\n{'='*60}")
    print(f"INSPECTING: {os.path.basename(filename)}")
    print(f"{'='*60}")
    
    try:
        if filename.endswith('.gz'):
            with gzip.open(filename, 'rb') as f:
                chunk_data = f.read()
        else:
            with open(filename, 'rb') as f:
                chunk_data = f.read()
                
        print(f"Compressed size: {os.path.getsize(filename):,} bytes")
        print(f"Uncompressed size: {len(chunk_data):,} bytes")
        
        # Determine version and record size
        version = chunk_data[0:4]
        version_int = struct.unpack('i', version)[0]
        print(f"Version: {version_int}")
        
        v6_struct = struct.Struct(V6_STRUCT_STRING)
        record_size = v6_struct.size
        num_records = len(chunk_data) // record_size
        
        print(f"Record size: {record_size} bytes")
        print(f"Number of records: {num_records}")
        
        # Inspect first few records
        records_inspected = 0
        for i in range(0, min(len(chunk_data), max_records * record_size), record_size):
            record = chunk_data[i:i + record_size]
            
            if len(record) == record_size:
                print(f"\n--- RECORD {records_inspected + 1} ---")
                if inspect_single_record(record):
                    records_inspected += 1
                    if records_inspected >= max_records:
                        break
                        
    except Exception as e:
        print(f"Error reading file: {e}")

def analyze_dataset_overview(data_path, max_files=10):
    """Analyze overview of entire dataset"""
    chunk_files = glob.glob(os.path.join(data_path, "*.gz"))
    
    print(f"DATASET OVERVIEW")
    print(f"{'='*60}")
    print(f"Data path: {data_path}")
    print(f"Total chunk files: {len(chunk_files)}")
    
    if not chunk_files:
        print("No .gz files found!")
        return
    
    # Sort by filename for consistent ordering
    chunk_files.sort()
    
    total_size = sum(os.path.getsize(f) for f in chunk_files)
    print(f"Total compressed size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    
    # File size statistics
    sizes = [os.path.getsize(f) for f in chunk_files]
    print(f"File size range: {min(sizes):,} - {max(sizes):,} bytes")
    print(f"Average file size: {sum(sizes)/len(sizes):.0f} bytes")
    
    # Sample a few files
    sample_files = chunk_files[:max_files] if len(chunk_files) >= max_files else chunk_files
    
    for filename in sample_files:
        inspect_chunk_file(filename, max_records=2)

def test_pytorch_loading(data_path, max_files=10):
    """Test the PyTorch dataset loading"""
    print(f"\nPYTORCH DATASET LOADING TEST")
    print(f"{'='*60}")
    
    chunk_files = glob.glob(os.path.join(data_path, "*.gz"))[:max_files]
    
    if not chunk_files:
        print("No chunk files found!")
        return
        
    try:
        dataset = ChessDataset(chunk_files, sample_rate=1)
        print(f"Dataset loaded successfully!")
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test first sample
            planes, policy, value, best_q, moves_left = dataset[0]
            
            print(f"\nFirst sample:")
            print(f"  Planes shape: {planes.shape}")
            print(f"  Policy shape: {policy.shape}")
            print(f"  Value shape: {value.shape}")
            print(f"  Best Q shape: {best_q.shape}")
            print(f"  Moves left shape: {moves_left.shape}")
            
            print(f"\nValue breakdown (WDL):")
            print(f"  Win: {value[0]:.4f}")
            print(f"  Draw: {value[1]:.4f}")
            print(f"  Loss: {value[2]:.4f}")
            print(f"  Sum: {value.sum():.4f}")
            
            print(f"\nPolicy stats:")
            print(f"  Policy sum: {policy.sum():.6f}")
            print(f"  Max policy: {policy.max():.6f}")
            print(f"  Non-zero moves: {(policy > 0).sum().item()}")
            
            print(f"\nBoard representation:")
            print(f"  Total active squares: {(planes > 0).sum().item()}")
            print(f"  Planes with any pieces: {(planes.sum(dim=[1,2]) > 0).sum().item()}/112")
            
            # Show first few planes that have pieces
            for i in range(min(8, planes.shape[0])):
                pieces = (planes[i] > 0).sum().item()
                if pieces > 0:
                    print(f"    Plane {i}: {pieces} active squares")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

def main():
    # data_path = "/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run3--20210605-0521"
    data_path = "/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run1--20250209-1017"
    print("LCZERO TRAINING DATA INSPECTOR")
    print("=" * 80)
    
    # 1. Dataset overview
    analyze_dataset_overview(data_path, max_files=5)
    
    # 2. Test PyTorch loading
    test_pytorch_loading(data_path, max_files=5)

if __name__ == "__main__":
    main()