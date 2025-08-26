"""
Visualize chess dataset created by build_chess_dataset.py.
Shows dataset structure, statistics, and sample data.
Updated for new format with proper move encoding and move masking.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from collections import Counter
import argparse
import sys

# Add parent directory to path to access dataset modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import chess move encoder from build script
try:
    from dataset.build_chess_dataset import ChessMoveEncoder
    import chess
except ImportError:
    print("Warning: Could not import chess modules. Some features may not work.")
    ChessMoveEncoder = None
    chess = None


class ChessDatasetVisualizer:
    """Visualize chess dataset files and statistics."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = os.path.abspath(dataset_path)
        self.train_path = os.path.join(self.dataset_path, "train")
        self.test_path = os.path.join(self.dataset_path, "test")
        
        # Check if dataset path exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        print(f"Initializing visualizer for dataset: {self.dataset_path}")
        
        # Initialize move encoder if available
        if ChessMoveEncoder:
            try:
                self.move_encoder = ChessMoveEncoder()
                print("✅ Chess move encoder loaded successfully")
            except Exception as e:
                print(f"⚠️  Could not initialize move encoder: {e}")
                self.move_encoder = None
        else:
            print("⚠️  Chess move encoder not available")
            self.move_encoder = None
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.train_metadata = self._load_split_metadata("train")
        self.test_metadata = self._load_split_metadata("test")
        
        # Load dataset arrays
        print("\nLoading dataset arrays...")
        self.train_data = self._load_split_data("train")
        self.test_data = self._load_split_data("test")
        
        # Summary
        train_examples = len(self.train_data.get('inputs', []))
        test_examples = len(self.test_data.get('inputs', []))
        print(f"\nDataset loaded: {train_examples:,} train + {test_examples:,} test examples")
    
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
            print(f"Warning: {split} directory not found at {split_path}")
            return data
        
        # Load all .npy files
        npy_files = [f for f in os.listdir(split_path) if f.endswith('.npy')]
        if not npy_files:
            print(f"Warning: No .npy files found in {split_path}")
            return data
        
        for filename in npy_files:
            field_name = filename.replace('all__', '').replace('.npy', '')
            filepath = os.path.join(split_path, filename)
            try:
                array = np.load(filepath, allow_pickle=True)
                data[field_name] = array
                print(f"Loaded {split}/{field_name}: {array.shape} {array.dtype}")
            except Exception as e:
                print(f"Error loading {split}/{filename}: {e}")
        
        return data
    
    def print_dataset_overview(self):
        """Print dataset overview and statistics."""
        print("=" * 60)
        print("CHESS DATASET OVERVIEW")
        print("=" * 60)
        
        # Main metadata
        print("\n📊 Dataset Metadata:")
        for key, value in self.metadata.items():
            if key == "elo_ranges":
                print(f"  {key}: {len(value)} ELO groups")
                for i, (min_elo, max_elo) in enumerate(value):
                    print(f"    Group {i}: ELO {min_elo}-{max_elo}")
            elif key == "groups":
                print(f"  {key}: {len(value)} difficulty groups")
            else:
                print(f"  {key}: {value}")
        
        # Split information
        train_size = len(self.train_data.get('inputs', []))
        test_size = len(self.test_data.get('inputs', []))
        total_size = train_size + test_size
        
        print(f"\n📂 Dataset Splits:")
        print(f"  Train: {train_size:,} examples")
        print(f"  Test:  {test_size:,} examples")
        print(f"  Total: {total_size:,} examples")
        
        if total_size > 0:
            print(f"  Split ratio: {train_size/total_size:.2f} / {test_size/total_size:.2f}")
        else:
            print("  Split ratio: No data found")
        
        # File structure
        print(f"\n📁 File Structure:")
        for split in ["train", "test"]:
            split_path = os.path.join(self.dataset_path, split)
            if os.path.exists(split_path):
                print(f"  {split}/")
                for filename in sorted(os.listdir(split_path)):
                    filepath = os.path.join(split_path, filename)
                    if filename.endswith('.npy'):
                        try:
                            array = np.load(filepath, allow_pickle=True)
                            size_mb = os.path.getsize(filepath) / (1024 * 1024)
                            print(f"    {filename:<30} shape: {str(array.shape):<15} dtype: {str(array.dtype):<10} size: {size_mb:.1f}MB")
                        except:
                            size_mb = os.path.getsize(filepath) / (1024 * 1024)
                            print(f"    {filename:<30} {'ERROR':<15} {'N/A':<10} size: {size_mb:.1f}MB")
                    else:
                        size_kb = os.path.getsize(filepath) / 1024
                        print(f"    {filename:<30} {'N/A':<15} {'N/A':<10} size: {size_kb:.1f}KB")
    
    def analyze_move_distribution(self, split: str = "train") -> Tuple[Counter, List[str]]:
        """Analyze distribution of chess moves."""
        data = self.train_data if split == "train" else self.test_data
        move_targets = data.get('move_targets', np.array([]))
        
        if len(move_targets) == 0:
            return Counter(), []
        
        # Count move frequencies
        move_counter = Counter(move_targets)
        
        # Decode most common moves if encoder is available
        top_moves = []
        for move_id, count in move_counter.most_common(20):
            if self.move_encoder:
                try:
                    move_str = self.move_encoder.decode_move(move_id)
                    top_moves.append(f"{move_str} ({count} times)")
                except:
                    top_moves.append(f"move_{move_id} ({count} times)")
            else:
                top_moves.append(f"move_{move_id} ({count} times)")
        
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
    
    def analyze_move_masks(self, split: str = "train") -> Dict[str, Any]:
        """Analyze move mask statistics."""
        data = self.train_data if split == "train" else self.test_data
        move_masks = data.get('possible_moves', np.array([]))
        
        if len(move_masks) == 0:
            return {}
        
        # Calculate statistics
        valid_moves_per_position = np.sum(move_masks, axis=1)
        
        analysis = {
            'shape': move_masks.shape,
            'avg_valid_moves': np.mean(valid_moves_per_position),
            'std_valid_moves': np.std(valid_moves_per_position),
            'min_valid_moves': np.min(valid_moves_per_position),
            'max_valid_moves': np.max(valid_moves_per_position),
            'total_possible_actions': move_masks.shape[1],
            'mask_sparsity': 1 - np.mean(move_masks)  # Fraction of invalid moves
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
        move_names = []
        for mid in move_ids:
            if self.move_encoder:
                try:
                    move_names.append(self.move_encoder.decode_move(mid))
                except:
                    move_names.append(f"move_{mid}")
            else:
                move_names.append(f"move_{mid}")
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(move_names)), counts, alpha=0.8)
        plt.title(f'Top {top_n} Most Frequent Moves ({split.title()} Set)', fontsize=16)
        plt.xlabel('Chess Moves', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(range(len(move_names)), move_names, rotation=45, ha='right')
        
        # Color bars by frequency
        max_count = max(counts)
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(counts[i] / max_count))
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_count*0.01, 
                    str(count), ha='center', va='bottom', fontsize=8)
        
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
        axes[0, 0].hist(non_zero_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Non-Zero Tokens per Position')
        axes[0, 0].set_xlabel('Number of Non-Zero Tokens')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(non_zero_counts), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(non_zero_counts):.1f}')
        axes[0, 0].legend()
        
        # 2. Token value distribution
        token_values = inputs[inputs > 0]  # Only non-zero tokens
        if len(token_values) > 0:
            axes[0, 1].hist(token_values, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Token Value Distribution')
            axes[0, 1].set_xlabel('Token Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_yscale('log')
        
        # 3. Position sequence analysis
        if len(inputs) > 0:
            # Sample for performance
            sample_size = min(1000, len(inputs))
            sample_indices = np.random.choice(len(inputs), sample_size, replace=False)
            sample_inputs = inputs[sample_indices]
            
            # Calculate effective lengths (non-padded)
            effective_lengths = []
            for seq in sample_inputs:
                # Find last non-zero token
                non_zero_positions = np.where(seq != 0)[0]
                if len(non_zero_positions) > 0:
                    effective_lengths.append(non_zero_positions[-1] + 1)
                else:
                    effective_lengths.append(0)
            
            axes[1, 0].hist(effective_lengths, bins=20, alpha=0.7, color='salmon', edgecolor='black')
            axes[1, 0].set_title(f'Effective Sequence Lengths (Sample of {sample_size})')
            axes[1, 0].set_xlabel('Sequence Length')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. Token position heat map
        if len(inputs) > 0:
            # Average token presence by position
            token_positions = np.mean(inputs > 0, axis=0)
            x = range(len(token_positions))
            axes[1, 1].bar(x, token_positions, color='orange', alpha=0.7, width=1.0)
            axes[1, 1].set_title('Token Presence by Sequence Position')
            axes[1, 1].set_xlabel('Sequence Position')
            axes[1, 1].set_ylabel('Average Presence')
            
            # Highlight important positions
            max_presence = np.max(token_positions)
            axes[1, 1].axhline(max_presence * 0.5, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_move_mask_statistics(self, split: str = "train"):
        """Plot statistics about move masks."""
        analysis = self.analyze_move_masks(split)
        
        if not analysis:
            print(f"No move mask data found for {split} split")
            return
        
        data = self.train_data if split == "train" else self.test_data
        move_masks = data.get('possible_moves', np.array([]))
        
        if len(move_masks) == 0:
            return
        
        # Calculate valid moves per position
        valid_moves_per_position = np.sum(move_masks, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Move Mask Statistics ({split.title()} Set)', fontsize=16)
        
        # 1. Distribution of valid moves per position
        axes[0, 0].hist(valid_moves_per_position, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].set_title('Valid Moves per Position')
        axes[0, 0].set_xlabel('Number of Valid Moves')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(analysis['avg_valid_moves'], color='red', linestyle='--', 
                          label=f"Mean: {analysis['avg_valid_moves']:.1f}")
        axes[0, 0].legend()
        
        # 2. Move popularity across all positions
        move_popularity = np.sum(move_masks, axis=0)  # How often each move is valid
        top_valid_moves = np.argsort(move_popularity)[-50:]  # Top 50 most commonly valid moves
        
        axes[0, 1].bar(range(len(top_valid_moves)), move_popularity[top_valid_moves], 
                      alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Top 50 Most Commonly Valid Moves')
        axes[0, 1].set_xlabel('Move Index (sorted by popularity)')
        axes[0, 1].set_ylabel('Times Valid')
        
        # 3. Sparsity visualization
        sparsity_data = [analysis['mask_sparsity'], 1 - analysis['mask_sparsity']]
        labels = ['Invalid Moves', 'Valid Moves']
        colors = ['lightcoral', 'lightgreen']
        
        axes[1, 0].pie(sparsity_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Move Mask Sparsity')
        
        # 4. Statistics summary
        axes[1, 1].axis('off')
        stats_text = f"""Move Mask Statistics:
        
Total Positions: {analysis['shape'][0]:,}
Total Possible Actions: {analysis['total_possible_actions']:,}

Valid Moves per Position:
  • Average: {analysis['avg_valid_moves']:.1f}
  • Std Dev: {analysis['std_valid_moves']:.1f}
  • Min: {analysis['min_valid_moves']}
  • Max: {analysis['max_valid_moves']}

Sparsity: {analysis['mask_sparsity']:.1%}
(Fraction of moves that are invalid)
        """
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def show_sample_data(self, split: str = "train", n_samples: int = 5):
        """Show sample input/target pairs."""
        data = self.train_data if split == "train" else self.test_data
        
        inputs = data.get('inputs', np.array([]))
        move_targets = data.get('move_targets', np.array([]))
        puzzle_ids = data.get('puzzle_identifiers', np.array([]))
        move_masks = data.get('possible_moves', np.array([]))
        
        if len(inputs) == 0:
            print(f"No data found for {split} split")
            return
        
        print(f"\n🎯 Sample Data ({split.title()} Set):")
        print("=" * 80)
        
        n_samples = min(n_samples, len(inputs))
        
        for i in range(n_samples):
            print(f"\nSample {i + 1}:")
            print(f"  Puzzle ID: {puzzle_ids[i] if len(puzzle_ids) > i else 'N/A'}")
            
            # Show input sequence statistics
            input_seq = inputs[i]
            non_zero_count = np.count_nonzero(input_seq)
            print(f"  Input sequence: {len(input_seq)} tokens, {non_zero_count} non-zero")
            print(f"  Token range: [{np.min(input_seq)}, {np.max(input_seq)}]")
            
            # Show move mask info
            if len(move_masks) > i:
                valid_moves = np.sum(move_masks[i])
                print(f"  Valid moves: {valid_moves} out of {len(move_masks[i])}")
            
            # Show target move
            if len(move_targets) > i:
                if self.move_encoder:
                    try:
                        target_move = self.move_encoder.decode_move(move_targets[i])
                        print(f"  Target move: {target_move} (id: {move_targets[i]})")
                    except:
                        print(f"  Target move: move_{move_targets[i]} (id: {move_targets[i]})")
                else:
                    print(f"  Target move: move_{move_targets[i]} (id: {move_targets[i]})")
            
            print(f"  Input preview: {input_seq[:20]}...")
    
    def validate_dataset_integrity(self):
        """Validate dataset integrity and format."""
        print("\n🔍 Dataset Validation:")
        print("=" * 50)
        
        issues = []
        
        # Check if required files exist
        required_files = ["inputs", "labels", "move_targets", "puzzle_identifiers", 
                         "puzzle_indices", "group_indices", "possible_moves"]
        
        for split in ["train", "test"]:
            data = self.train_data if split == "train" else self.test_data
            print(f"\n{split.title()} Split:")
            
            for file_type in required_files:
                if file_type in data:
                    array = data[file_type]
                    print(f"  ✅ {file_type}: {array.shape} {array.dtype}")
                    
                    # Check for anomalies
                    if file_type == "move_targets":
                        max_valid_move = self.metadata.get('num_actions', 4352) - 1
                        invalid_moves = np.sum(array > max_valid_move)
                        if invalid_moves > 0:
                            issues.append(f"{split}: {invalid_moves} invalid move targets (> {max_valid_move})")
                    
                    elif file_type == "inputs":
                        expected_seq_len = self.metadata.get('seq_len', 200)
                        if array.shape[1] != expected_seq_len:
                            issues.append(f"{split}: input sequence length mismatch ({array.shape[1]} vs {expected_seq_len})")
                    
                    elif file_type == "possible_moves":
                        expected_actions = self.metadata.get('num_actions', 4352)
                        if len(array.shape) == 2 and array.shape[1] != expected_actions:
                            issues.append(f"{split}: move mask action count mismatch ({array.shape[1]} vs {expected_actions})")
                else:
                    print(f"  ❌ {file_type}: MISSING")
                    issues.append(f"{split}: missing {file_type}")
        
        # Check data consistency
        for split in ["train", "test"]:
            data = self.train_data if split == "train" else self.test_data
            if not data:
                continue
            
            # Main arrays should have same first dimension
            main_arrays = ["inputs", "labels", "move_targets", "puzzle_identifiers", "possible_moves"]
            lengths = {}
            for k in main_arrays:
                if k in data:
                    lengths[k] = len(data[k])
            
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
                
                # Move mask analysis
                mask_analysis = self.analyze_move_masks(split)
                if mask_analysis:
                    print(f"  Avg valid moves per position: {mask_analysis['avg_valid_moves']:.1f}")
                    print(f"  Move mask sparsity: {mask_analysis['mask_sparsity']:.1%}")
        
        # Sample data
        self.show_sample_data("train", 3)
        if self.test_data:
            self.show_sample_data("test", 2)


def main():
    parser = argparse.ArgumentParser(description="Visualize chess dataset")
    parser.add_argument("--dataset-path", default="../data/chess-move-prediction",
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
                
                # New: move mask plots
                if f'possible_moves' in (visualizer.train_data if split == "train" else visualizer.test_data):
                    visualizer.plot_move_mask_statistics(split)


if __name__ == "__main__":
    main()