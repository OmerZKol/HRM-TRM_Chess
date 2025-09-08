"""
Test script to verify chess_puzzle_dataset.py instantiation.
"""

import os
import sys
import traceback
import torch
from typing import Dict, Any

# Add current directory to path
sys.path.append('.')

def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    
    try:
        import numpy as np
        print("✅ numpy available")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import torch
        print("✅ torch available")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import pydantic
        print("✅ pydantic available")
    except ImportError:
        missing_deps.append("pydantic")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        return False
    
    return True


def check_dataset_files():
    """Check if dataset files exist."""
    print("\n🔍 Checking dataset files...")
    
    dataset_path = "data/chess-move-prediction"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory doesn't exist: {dataset_path}")
        return False
    
    print(f"✅ Dataset directory exists: {dataset_path}")
    
    # Check required files
    required_files = [
        "dataset.json",
        "train/dataset.json",
        "test/dataset.json",
        "train/all__inputs.npy",
        "train/all__labels.npy", 
        "train/all__move_targets.npy",
        "train/all__puzzle_identifiers.npy",
        "train/all__puzzle_indices.npy",
        "train/all__group_indices.npy",
        "test/all__inputs.npy",
        "test/all__labels.npy",
        "test/all__move_targets.npy",
        "test/all__puzzle_identifiers.npy",
        "test/all__puzzle_indices.npy",
        "test/all__group_indices.npy"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(dataset_path, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print(f"\nPlease run: python dataset/build_chess_dataset.py")
        return False
    
    return True


def test_imports():
    """Test if we can import the required modules."""
    print("\n🔍 Testing imports...")
    
    try:
        from models.losses import IGNORE_LABEL_ID
        print("✅ Successfully imported IGNORE_LABEL_ID")
    except Exception as e:
        print(f"❌ Failed to import IGNORE_LABEL_ID: {e}")
        return False
    
    try:
        from dataset.common import PuzzleDatasetMetadata
        print("✅ Successfully imported PuzzleDatasetMetadata")
    except Exception as e:
        print(f"❌ Failed to import PuzzleDatasetMetadata: {e}")
        return False
    
    try:
        from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, _sample_batch
        print("✅ Successfully imported puzzle_dataset components")
    except Exception as e:
        print(f"❌ Failed to import puzzle_dataset components: {e}")
        traceback.print_exc()
        return False
    
    try:
        from chess_puzzle_dataset import ChessPuzzleDataset, ChessPuzzleDatasetConfig
        print("✅ Successfully imported ChessPuzzleDataset classes")
        return True
    except Exception as e:
        print(f"❌ Failed to import chess dataset classes: {e}")
        traceback.print_exc()
        return False


def test_chess_dataset_instantiation():
    """Test if ChessPuzzleDataset can be instantiated correctly."""
    
    print("=" * 60)
    print("TESTING CHESS PUZZLE DATASET INSTANTIATION")
    print("=" * 60)
    
    try:
        # Import the chess dataset
        from chess_puzzle_dataset import ChessPuzzleDataset, ChessPuzzleDatasetConfig
        print("✅ Successfully imported ChessPuzzleDataset classes")
    except Exception as e:
        print(f"❌ Failed to import chess dataset classes: {e}")
        traceback.print_exc()
        return False
    
    # Test configuration
    dataset_path = "data/chess-move-prediction"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        print("Please run build_chess_dataset.py first to create the dataset.")
        return False
    
    print(f"✅ Dataset path exists: {dataset_path}")
    
    # Check for train/test splits
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")
    
    if not os.path.exists(train_path):
        print(f"❌ Train split not found: {train_path}")
        return False
    
    if not os.path.exists(test_path):
        print(f"❌ Test split not found: {test_path}")
        return False
    
    print("✅ Train and test splits found")
    
    # Test dataset configuration
    try:
        config = ChessPuzzleDatasetConfig(
            seed=42,
            dataset_path=dataset_path,
            global_batch_size=32,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        print("✅ ChessPuzzleDatasetConfig created successfully")
        print(f"   Config: {config}")
    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        traceback.print_exc()
        return False
    
    # Test train dataset instantiation
    try:
        print("\n📂 Testing train dataset instantiation...")
        train_dataset = ChessPuzzleDataset(config, split="train")
        print("✅ Train dataset instantiated successfully")
        
        # Check metadata
        if hasattr(train_dataset, 'metadata'):
            print(f"   Metadata: {train_dataset.metadata}")
        else:
            print("⚠️  No metadata attribute found")
        
    except Exception as e:
        print(f"❌ Failed to instantiate train dataset: {e}")
        traceback.print_exc()
        return False
    
    # Test test dataset instantiation
    try:
        print("\n📂 Testing test dataset instantiation...")
        test_config = ChessPuzzleDatasetConfig(
            seed=42,
            dataset_path=dataset_path,
            global_batch_size=32,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        test_dataset = ChessPuzzleDataset(test_config, split="test")
        print("✅ Test dataset instantiated successfully")
        
    except Exception as e:
        print(f"❌ Failed to instantiate test dataset: {e}")
        traceback.print_exc()
        return False
    
    # Test data loading
    try:
        print("\n🔄 Testing data loading...")
        
        # Force data loading by accessing _data
        train_dataset._lazy_load_dataset()
        
        if train_dataset._data is None:
            print("❌ Data not loaded (still None)")
            return False
        
        print("✅ Data loaded successfully")
        
        # Check data structure
        for set_name, dataset in train_dataset._data.items():
            print(f"   Set '{set_name}':")
            for field_name, array in dataset.items():
                print(f"     {field_name}: shape={array.shape}, dtype={array.dtype}")
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        traceback.print_exc()
        return False
    
    # Test batch iteration
    try:
        print("\n🔄 Testing batch iteration...")
        
        batch_count = 0
        for set_name, batch, batch_size in train_dataset:
            print(f"✅ Got batch {batch_count + 1}:")
            print(f"   Set name: {set_name}")
            print(f"   Batch size: {batch_size}")
            print(f"   Batch keys: {list(batch.keys())}")
            
            # Check batch contents
            for key, tensor in batch.items():
                if torch.is_tensor(tensor):
                    print(f"   {key}: shape={tensor.shape}, dtype={tensor.dtype}")
                else:
                    print(f"   {key}: {type(tensor)} = {tensor}")
            
            # Check for required fields
            required_fields = ["inputs", "labels", "puzzle_identifiers"]
            chess_fields = ["move_targets"]
            
            for field in required_fields:
                if field not in batch:
                    print(f"⚠️  Missing required field: {field}")
                else:
                    print(f"   ✅ Has {field}")
            
            for field in chess_fields:
                if field not in batch:
                    print(f"⚠️  Missing chess-specific field: {field}")
                else:
                    print(f"   ✅ Has {field}")
            
            batch_count += 1
            if batch_count >= 2:  # Test first 2 batches
                break
                
        print(f"✅ Successfully iterated {batch_count} batches")
        
    except Exception as e:
        print(f"❌ Failed to iterate batches: {e}")
        traceback.print_exc()
        return False
    
    # Test test dataset iteration
    try:
        print("\n🔄 Testing test dataset iteration...")
        
        batch_count = 0
        for set_name, batch, batch_size in test_dataset:
            print(f"✅ Test batch {batch_count + 1}: size={batch_size}, keys={list(batch.keys())}")
            batch_count += 1
            if batch_count >= 1:  # Test first batch
                break
                
        print(f"✅ Successfully iterated test dataset")
        
    except Exception as e:
        print(f"❌ Failed to iterate test dataset: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("ChessPuzzleDataset instantiates correctly")
    print("=" * 60)
    
    return True


def test_dataset_compatibility():
    """Test compatibility with training script expectations."""
    
    print("\n🔄 Testing training script compatibility...")
    
    try:
        from chess_puzzle_dataset import ChessPuzzleDataset, ChessPuzzleDatasetConfig
        from torch.utils.data import DataLoader
        
        # Create dataset like in training script
        dataset_config = ChessPuzzleDatasetConfig(
            seed=42,
            dataset_path="data/chess-move-prediction",
            global_batch_size=8,  # Small batch for testing
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        train_dataset = ChessPuzzleDataset(dataset_config, split="train")
        
        # Create DataLoader like in training script
        train_loader = DataLoader(
            train_dataset,
            batch_size=None,  # Already batched
            num_workers=0,    # Single-threaded
            pin_memory=True
        )
        
        print("✅ DataLoader created successfully")
        
        # Test iteration like in training script
        for step_idx, (set_name, batch, batch_size) in enumerate(train_loader):
            print(f"✅ Training-style iteration {step_idx + 1}:")
            print(f"   set_name={set_name}, batch_size={batch_size}")
            print(f"   batch keys: {list(batch.keys())}")
            
            # Check tensor properties
            for k, v in batch.items():
                if hasattr(v, 'shape'):
                    print(f"   {k}: shape={v.shape}, dtype={v.dtype}")
            
            if step_idx >= 1:  # Test first 2 iterations
                break
        
        print("✅ Training-style iteration successful")
        return True
        
    except Exception as e:
        print(f"❌ Training compatibility test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("CHESS DATASET INSTANTIATION TEST")
    print("=" * 60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Dependencies check failed")
        sys.exit(1)
    
    # Step 2: Check dataset files
    if not check_dataset_files():
        print("❌ Dataset files check failed")
        sys.exit(1)
    
    # Step 3: Test imports
    if not test_imports():
        print("❌ Import test failed")
        sys.exit(1)
    
    # Step 4: Test dataset instantiation
    success = test_chess_dataset_instantiation()
    
    # Step 5: Test training compatibility
    if success:
        test_dataset_compatibility()
    
    print(f"\n{'✅ ALL TESTS PASSED' if success else '❌ TESTS FAILED'}")
