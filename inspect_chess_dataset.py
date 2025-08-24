"""
Quick chess dataset inspector - text-only version without plotting dependencies.
"""

import os
import json
import numpy as np
from collections import Counter
import sys

# Add dataset directory to path
sys.path.append('dataset')

def inspect_chess_dataset(dataset_path: str):
    """Quick inspection of chess dataset files."""
    
    print("=" * 60)
    print("CHESS DATASET QUICK INSPECTION")
    print("=" * 60)
    
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    # Load main metadata
    metadata_path = os.path.join(dataset_path, "dataset.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("\n📊 Main Dataset Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("❌ Main dataset.json not found")
    
    # Check each split
    for split in ["train", "test"]:
        split_path = os.path.join(dataset_path, split)
        print(f"\n📂 {split.title()} Split:")
        
        if not os.path.exists(split_path):
            print(f"  ❌ {split} directory not found")
            continue
        
        # List files
        files = os.listdir(split_path)
        npy_files = [f for f in files if f.endswith('.npy')]
        
        print(f"  📁 Files found: {len(files)} total, {len(npy_files)} .npy files")
        
        # Load and inspect each .npy file
        for filename in sorted(npy_files):
            filepath = os.path.join(split_path, filename)
            try:
                array = np.load(filepath)
                field_name = filename.replace('all__', '').replace('.npy', '')
                
                print(f"    📄 {field_name}:")
                print(f"      Shape: {array.shape}")
                print(f"      Dtype: {array.dtype}")
                print(f"      Range: [{np.min(array)}, {np.max(array)}]")
                
                # Special analysis for different field types
                if field_name == "inputs":
                    non_zero_ratio = np.count_nonzero(array) / array.size
                    print(f"      Non-zero ratio: {non_zero_ratio:.3f}")
                    
                elif field_name == "move_targets":
                    unique_moves = len(np.unique(array))
                    print(f"      Unique moves: {unique_moves}")
                    
                    # Show most common moves
                    counter = Counter(array)
                    top_5 = counter.most_common(5)
                    print(f"      Top moves: {top_5}")
                
                elif field_name == "puzzle_identifiers":
                    unique_puzzles = len(np.unique(array))
                    print(f"      Unique puzzles: {unique_puzzles}")
                
                print()
                
            except Exception as e:
                print(f"    ❌ Error loading {filename}: {e}")
        
        # Load split metadata
        split_metadata_path = os.path.join(split_path, "dataset.json")
        if os.path.exists(split_metadata_path):
            with open(split_metadata_path, 'r') as f:
                split_metadata = json.load(f)
            print(f"  📋 {split.title()} Metadata:")
            key_fields = ["size", "vocab_size", "seq_len", "num_actions"]
            for key in key_fields:
                if key in split_metadata:
                    print(f"    {key}: {split_metadata[key]}")
    
    # Dataset integrity check
    print("\n🔍 Basic Integrity Check:")
    
    issues = []
    for split in ["train", "test"]:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
        
        # Expected files
        expected_files = [
            "all__inputs.npy",
            "all__labels.npy", 
            "all__move_targets.npy",
            "all__puzzle_identifiers.npy",
            "all__puzzle_indices.npy",
            "all__group_indices.npy"
        ]
        
        missing_files = []
        array_shapes = {}
        
        for filename in expected_files:
            filepath = os.path.join(split_path, filename)
            if os.path.exists(filepath):
                try:
                    array = np.load(filepath)
                    array_shapes[filename] = array.shape[0]  # First dimension
                except:
                    issues.append(f"{split}: Could not load {filename}")
            else:
                missing_files.append(filename)
        
        if missing_files:
            issues.append(f"{split}: Missing files: {missing_files}")
        
        # Check shape consistency
        main_arrays = ["all__inputs.npy", "all__labels.npy", "all__move_targets.npy", "all__puzzle_identifiers.npy"]
        main_shapes = {k: v for k, v in array_shapes.items() if k in main_arrays}
        
        if len(set(main_shapes.values())) > 1:
            issues.append(f"{split}: Inconsistent array lengths: {main_shapes}")
    
    if issues:
        print("  ⚠️  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ Basic integrity check passed!")
    
    print(f"\n✨ Inspection complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick chess dataset inspection")
    parser.add_argument("--dataset-path", default="data/chess-move-prediction",
                       help="Path to chess dataset directory")
    
    args = parser.parse_args()
    inspect_chess_dataset(args.dataset_path)
