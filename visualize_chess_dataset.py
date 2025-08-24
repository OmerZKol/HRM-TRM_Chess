"""
Visualize chess dataset created by build_chess_dataset.py.
Shows dataset structure, statistics, and sample data.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from collections import Counter
import argparse

# Import chess move encoder from build script
import sys
sys.path.append('dataset')
from dataset.build_chess_dataset import ChessMoveEncoder, IDX_TO_SQUARE


class ChessDatasetVisualizer:
    """Visualize chess dataset files and statistics."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.train_path = os.path.join(dataset_path, "train")
        self.test_path = os.path.join(dataset_path, "test")
        self.move_encoder = ChessMoveEncoder()
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.train_metadata = self._load_split_metadata("train")
        self.test_metadata = self._load_split_metadata("test")
        
        # Load dataset arrays
        self.train_data = self._load_split_data("train")
        self.test_data = self._load_split_data("test")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load main dataset metadata."""
        metadata_path = os.path.join(self.dataset_path, "dataset.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_split_metadata(self, split: str) -> Dict[str, Any]:
        """Load split-specific metadata."""
        metadata_path = os.path.join(self.dataset_path, split, "dataset.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_split_data(self, split: str) -> Dict[str, np.ndarray]:
        """Load all data arrays for a split."""
        split_path = os.path.join(self.dataset_path, split)
        data = {}
        
        if not os.path.exists(split_path):
            return data
        
        # Load all .npy files
        for filename in os.listdir(split_path):
            if filename.endswith('.npy'):
                field_name = filename.replace('all__', '').replace('.npy', '')
                filepath = os.path.join(split_path, filename)
                try:
                    data[field_name] = np.load(filepath)
                    print(f"Loaded {field_name}: {data[field_name].shape}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return data
    
    def print_dataset_overview(self):
        """Print dataset overview and statistics."""
        print("=" * 60)
        print("CHESS DATASET OVERVIEW")
        print("=" * 60)
        
        # Main metadata
        print("\n📊 Dataset Metadata:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")
        
        # Split information
        print(f"\n📂 Dataset Splits:")
        print(f"  Train: {len(self.train_data.get('inputs', []))} examples")
        print(f"  Test:  {len(self.test_data.get('inputs', []))} examples")
        print(f"  Total: {len(self.train_data.get('inputs', [])) + len(self.test_data.get('inputs', []))} examples")
        
        # File structure
        print(f"\n📁 File Structure:")
        for split in ["train", "test"]:
            split_path = os.path.join(self.dataset_path, split)
            if os.path.exists(split_path):
                print(f"  {split}/")
                for filename in sorted(os.listdir(split_path)):
                    filepath = os.path.join(split_path, filename)
                    if filename.endswith('.npy'):
                        array = np.load(filepath)
                        print(f"    {filename:<25} shape: {array.shape}, dtype: {array.dtype}")
                    else:
                        size = os.path.getsize(filepath)
                        print(f"    {filename:<25} size: {size} bytes")
    
    def analyze_move_distribution(self, split: str = "train") -> Tuple[Counter, List[str]]:
        """Analyze distribution of chess moves."""
        data = self.train_data if split == "train" else self.test_data
        move_targets = data.get('move_targets', np.array([]))
        
        if len(move_targets) == 0:
            return Counter(), []
        
        # Count move frequencies
        move_counter = Counter(move_targets)
        
        # Decode most common moves
        top_moves = []
        for move_id, count in move_counter.most_common(20):
            move_str = self.move_encoder.decode_move(move_id)
            top_moves.append(f"{move_str} ({count} times)")
        
        return move_counter, top_moves
    
    def analyze_position_patterns(self, split: str = "train") -> Dict[str, Any]:
        """Analyze input position patterns."""
        data = self.train_data if split == "train" else self.test_data
        inputs = data.get('inputs', np.array([]))
        
        if len(inputs) == 0:
            return {}
        
        # Analyze sequence lengths and patterns
        analysis = {
            'shape': inputs.shape,
            'non_zero_counts': np.count_nonzero(inputs, axis=1),
            'unique_tokens': len(np.unique(inputs)),
            'max_token': np.max(inputs),
            'min_token': np.min(inputs),
            'mean_non_zero': np.mean(np.count_nonzero(inputs, axis=1))
        }
        
        return analysis
    
    def plot_move_distribution(self, split: str = "train", top_n: int = 20):
        """Plot distribution of chess moves."""
        move_counter, _ = self.analyze_move_distribution(split)
        
        if not move_counter:
            print(f"No move data found for {split} split")
            return
        
        # Get top moves
        top_moves = move_counter.most_common(top_n)
        move_ids, counts = zip(*top_moves)
        
        # Decode move names
        move_names = [self.move_encoder.decode_move(mid) for mid in move_ids]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(move_names)), counts)
        plt.title(f'Top {top_n} Most Frequent Moves ({split.title()} Set)')
        plt.xlabel('Chess Moves')
        plt.ylabel('Frequency')
        plt.xticks(range(len(move_names)), move_names, rotation=45, ha='right')
        
        # Color bars by frequency
        max_count = max(counts)
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(counts[i] / max_count))
        
        plt.tight_layout()
        plt.show()
    
    def plot_position_statistics(self, split: str = "train"):
        """Plot statistics about position encodings."""
        analysis = self.analyze_position_patterns(split)
        
        if not analysis:
            print(f"No input data found for {split} split")
            return
        
        data = self.train_data if split == "train" else self.test_data
        inputs = data.get('inputs', np.array([]))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Position Statistics ({split.title()} Set)', fontsize=16)
        
        # 1. Non-zero tokens per position
        non_zero_counts = analysis['non_zero_counts']
        axes[0, 0].hist(non_zero_counts, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Non-Zero Tokens per Position')
        axes[0, 0].set_xlabel('Number of Non-Zero Tokens')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Token value distribution
        token_values = inputs[inputs > 0]  # Only non-zero tokens
        axes[0, 1].hist(token_values, bins=50, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Token Value Distribution')
        axes[0, 1].set_xlabel('Token Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        
        # 3. Position length distribution (for sequences)
        seq_lengths = [len(np.trim_zeros(seq, 'f')) for seq in inputs[:1000]]  # Sample for speed
        axes[1, 0].hist(seq_lengths, bins=20, alpha=0.7, color='salmon')
        axes[1, 0].set_title('Effective Sequence Lengths (Sample)')
        axes[1, 0].set_xlabel('Sequence Length')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Heat map of token positions
        if len(inputs) > 0:
            token_positions = np.mean(inputs > 0, axis=0)  # Average token presence by position
            axes[1, 1].bar(range(len(token_positions)), token_positions, color='orange', alpha=0.7)
            axes[1, 1].set_title('Token Presence by Position')
            axes[1, 1].set_xlabel('Sequence Position')
            axes[1, 1].set_ylabel('Average Presence')
        
        plt.tight_layout()
        plt.show()
    
    def show_sample_data(self, split: str = "train", n_samples: int = 5):
        """Show sample input/target pairs."""
        data = self.train_data if split == "train" else self.test_data
        
        inputs = data.get('inputs', np.array([]))
        move_targets = data.get('move_targets', np.array([]))
        puzzle_ids = data.get('puzzle_identifiers', np.array([]))
        
        if len(inputs) == 0:
            print(f"No data found for {split} split")
            return
        
        print(f"\n🎯 Sample Data ({split.title()} Set):")
        print("=" * 80)
        
        n_samples = min(n_samples, len(inputs))
        
        for i in range(n_samples):
            print(f"\nSample {i + 1}:")
            print(f"  Puzzle ID: {puzzle_ids[i] if len(puzzle_ids) > i else 'N/A'}")
            
            # Show input sequence (decoded moves)
            input_seq = inputs[i]
            non_zero_moves = input_seq[input_seq > 0]
            decoded_moves = [self.move_encoder.decode_move(move_id) for move_id in non_zero_moves]
            print(f"  Position History: {' -> '.join(decoded_moves[-5:])}")  # Last 5 moves
            
            # Show target move
            if len(move_targets) > i:
                target_move = self.move_encoder.decode_move(move_targets[i])
                print(f"  Next Move Target: {target_move} (id: {move_targets[i]})")
            
            print(f"  Raw Input: {input_seq}")
    
    def validate_dataset_integrity(self):
        """Validate dataset integrity and format."""
        print("\n🔍 Dataset Validation:")
        print("=" * 50)
        
        issues = []
        
        # Check if required files exist
        required_files = ["inputs", "labels", "move_targets", "puzzle_identifiers", "puzzle_indices", "group_indices"]
        
        for split in ["train", "test"]:
            data = self.train_data if split == "train" else self.test_data
            print(f"\n{split.title()} Split:")
            
            for file_type in required_files:
                if file_type in data:
                    array = data[file_type]
                    print(f"  ✅ {file_type}: {array.shape} {array.dtype}")
                    
                    # Check for anomalies
                    if file_type == "move_targets":
                        invalid_moves = np.sum(array >= self.move_encoder.max_moves)
                        if invalid_moves > 0:
                            issues.append(f"{split}: {invalid_moves} invalid move targets")
                    
                    elif file_type == "inputs":
                        if array.shape[1] != self.metadata.get('seq_len', 10):
                            issues.append(f"{split}: input sequence length mismatch")
                else:
                    print(f"  ❌ {file_type}: MISSING")
                    issues.append(f"{split}: missing {file_type}")
        
        # Check data consistency
        for split in ["train", "test"]:
            data = self.train_data if split == "train" else self.test_data
            if not data:
                continue
                
            lengths = {k: len(v) for k, v in data.items() if k not in ["puzzle_indices", "group_indices"]}
            if len(set(lengths.values())) > 1:
                issues.append(f"{split}: inconsistent array lengths: {lengths}")
        
        if issues:
            print(f"\n⚠️  Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✅ Dataset validation passed!")
    
    def generate_report(self):
        """Generate comprehensive dataset report."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE CHESS DATASET REPORT")
        print("=" * 80)
        
        # Overview
        self.print_dataset_overview()
        
        # Validation
        self.validate_dataset_integrity()
        
        # Analysis for both splits
        for split in ["train", "test"]:
            if (split == "train" and self.train_data) or (split == "test" and self.test_data):
                print(f"\n📈 {split.title()} Split Analysis:")
                print("-" * 40)
                
                # Position analysis
                pos_analysis = self.analyze_position_patterns(split)
                for key, value in pos_analysis.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: mean={np.mean(value):.2f}, std={np.std(value):.2f}")
                    else:
                        print(f"  {key}: {value}")
                
                # Move analysis
                move_counter, top_moves = self.analyze_move_distribution(split)
                print(f"  Unique moves: {len(move_counter)}")
                print(f"  Top 5 moves: {top_moves[:5]}")
        
        # Sample data
        self.show_sample_data("train", 3)
        if self.test_data:
            self.show_sample_data("test", 2)


def main():
    parser = argparse.ArgumentParser(description="Visualize chess dataset")
    parser.add_argument("--dataset-path", default="data/chess-move-prediction",
                       help="Path to chess dataset directory")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate text report only (no plots)")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                       help="Which split to analyze")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path {args.dataset_path} does not exist!")
        return
    
    # Create visualizer
    visualizer = ChessDatasetVisualizer(args.dataset_path)
    
    # Generate report
    visualizer.generate_report()
    
    if not args.report_only:
        print("\n🎨 Generating visualizations...")
        
        # Generate plots for requested splits
        splits_to_plot = ["train", "test"] if args.split == "both" else [args.split]
        
        for split in splits_to_plot:
            if (split == "train" and visualizer.train_data) or (split == "test" and visualizer.test_data):
                print(f"\nPlotting {split} data...")
                visualizer.plot_move_distribution(split)
                visualizer.plot_position_statistics(split)


if __name__ == "__main__":
    main()
