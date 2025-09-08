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
                print("[OK] Chess move encoder loaded successfully")
            except Exception as e:
                print(f"[WARNING] Could not initialize move encoder: {e}")
                self.move_encoder = None
        else:
            print("[WARNING] Chess move encoder not available")
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
        
        # Check for value targets
        if 'value_targets' in self.train_data or 'value_targets' in self.test_data:
            print("[OK] Value targets found - enabling value analysis")
        else:
            print("[WARNING] No value targets found - value analysis disabled")
    
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
        print("\nDataset Metadata:")
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
        
        print(f"\nDataset Splits:")
        print(f"  Train: {train_size:,} examples")
        print(f"  Test:  {test_size:,} examples")
        print(f"  Total: {total_size:,} examples")
        
        if total_size > 0:
            print(f"  Split ratio: {train_size/total_size:.2f} / {test_size/total_size:.2f}")
        else:
            print("  Split ratio: No data found")
        
        # File structure
        print(f"\nFile Structure:")
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
        # For 3D inputs (batch, seq_len, features), count non-zero across all features per sequence position
        if len(inputs.shape) == 3:
            # Count non-zero tokens per position per sample, then sum across sequence positions
            non_zero_counts = np.count_nonzero(inputs, axis=2).sum(axis=1)
        else:
            # For 2D inputs, count non-zero per sequence
            non_zero_counts = np.count_nonzero(inputs, axis=1)
        
        analysis = {
            'shape': inputs.shape,
            'non_zero_counts': non_zero_counts,
            'unique_tokens': len(np.unique(inputs)),
            'max_token': np.max(inputs),
            'min_token': np.min(inputs),
            'mean_non_zero': np.mean(non_zero_counts)
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
    
    def analyze_value_targets(self, split: str = "train") -> Dict[str, Any]:
        """Analyze value target distribution and statistics."""
        data = self.train_data if split == "train" else self.test_data
        value_targets = data.get('value_targets', np.array([]))
        
        if len(value_targets) == 0:
            return {}
        
        # Filter out NaN values for analysis
        valid_values = value_targets[~np.isnan(value_targets)]
        
        if len(valid_values) == 0:
            return {'all_nan': True}
        
        analysis = {
            'total_samples': len(value_targets),
            'valid_samples': len(valid_values),
            'nan_count': len(value_targets) - len(valid_values),
            'nan_ratio': (len(value_targets) - len(valid_values)) / len(value_targets),
            'mean': np.mean(valid_values),
            'std': np.std(valid_values),
            'min': np.min(valid_values),
            'max': np.max(valid_values),
            'median': np.median(valid_values),
            'q25': np.percentile(valid_values, 25),
            'q75': np.percentile(valid_values, 75),
        }
        
        # Value distribution by sign (winning/losing perspective)
        positive_values = valid_values[valid_values > 0]
        negative_values = valid_values[valid_values < 0]
        zero_values = valid_values[valid_values == 0]
        
        analysis.update({
            'positive_count': len(positive_values),
            'negative_count': len(negative_values),
            'zero_count': len(zero_values),
            'positive_ratio': len(positive_values) / len(valid_values) if len(valid_values) > 0 else 0,
            'negative_ratio': len(negative_values) / len(valid_values) if len(valid_values) > 0 else 0,
            'zero_ratio': len(zero_values) / len(valid_values) if len(valid_values) > 0 else 0,
        })
        
        if len(positive_values) > 0:
            analysis['positive_mean'] = np.mean(positive_values)
            analysis['positive_std'] = np.std(positive_values)
        
        if len(negative_values) > 0:
            analysis['negative_mean'] = np.mean(negative_values)
            analysis['negative_std'] = np.std(negative_values)
        
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
                # For 3D inputs, check if any feature is non-zero at each position
                if len(seq.shape) == 2:  # (seq_len, features)
                    has_content = np.any(seq != 0, axis=1)  # (seq_len,)
                    non_zero_positions = np.where(has_content)[0]
                else:  # 1D sequence
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
            if len(inputs.shape) == 3:
                # For 3D inputs (batch, seq_len, features), average presence across batch and features
                token_positions = np.mean(np.mean(inputs > 0, axis=0), axis=1)  # (seq_len,)
            else:
                # For 2D inputs (batch, seq_len), average presence across batch
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
    
    def plot_value_target_distribution(self, split: str = "train"):
        """Plot comprehensive value target analysis."""
        analysis = self.analyze_value_targets(split)
        
        if not analysis or analysis.get('all_nan', False):
            print(f"No valid value target data found for {split} split")
            return
        
        data = self.train_data if split == "train" else self.test_data
        value_targets = data.get('value_targets', np.array([]))
        valid_values = value_targets[~np.isnan(value_targets)]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Value Target Analysis ({split.title()} Set)', fontsize=16)
        
        # 1. Value distribution histogram
        axes[0, 0].hist(valid_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Value Target Distribution')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(analysis['mean'], color='red', linestyle='--', 
                          label=f"Mean: {analysis['mean']:.3f}")
        axes[0, 0].axvline(analysis['median'], color='orange', linestyle='--', 
                          label=f"Median: {analysis['median']:.3f}")
        axes[0, 0].legend()
        
        # 2. Box plot for quartile analysis
        axes[0, 1].boxplot(valid_values, vert=True)
        axes[0, 1].set_title('Value Target Box Plot')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sign distribution (win/lose/draw)
        sign_counts = [analysis['negative_count'], analysis['zero_count'], analysis['positive_count']]
        sign_labels = ['Negative\\n(Losing)', 'Zero\\n(Draw)', 'Positive\\n(Winning)']
        colors = ['lightcoral', 'lightgray', 'lightgreen']
        
        axes[0, 2].bar(sign_labels, sign_counts, color=colors, alpha=0.8, edgecolor='black')
        axes[0, 2].set_title('Value Sign Distribution')
        axes[0, 2].set_ylabel('Count')
        
        # Add percentages on bars
        total = sum(sign_counts)
        for i, (count, label) in enumerate(zip(sign_counts, sign_labels)):
            if count > 0:
                pct = 100 * count / total
                axes[0, 2].text(i, count + total*0.01, f'{pct:.1f}%', 
                               ha='center', va='bottom', fontweight='bold')
        
        # 4. Value range analysis
        if analysis['positive_count'] > 0 and analysis['negative_count'] > 0:
            positive_values = valid_values[valid_values > 0]
            negative_values = valid_values[valid_values < 0]
            
            axes[1, 0].hist([negative_values, positive_values], bins=30, alpha=0.7, 
                           color=['lightcoral', 'lightgreen'], 
                           label=['Negative Values', 'Positive Values'],
                           edgecolor='black')
            axes[1, 0].set_title('Positive vs Negative Value Distribution')
            axes[1, 0].set_xlabel('Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].axvline(0, color='black', linestyle='-', alpha=0.5)
        else:
            axes[1, 0].hist(valid_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_title('Value Distribution (Single Sign)')
            axes[1, 0].set_xlabel('Value')
            axes[1, 0].set_ylabel('Frequency')
        
        # 5. Data completeness pie chart
        completeness_data = [analysis['valid_samples'], analysis['nan_count']]
        completeness_labels = ['Valid Values', 'NaN Values']
        completeness_colors = ['lightgreen', 'lightcoral']
        
        axes[1, 1].pie(completeness_data, labels=completeness_labels, colors=completeness_colors, 
                       autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Data Completeness')
        
        # 6. Statistics summary text
        axes[1, 2].axis('off')
        stats_text = f"""Value Target Statistics:
        
Total Samples: {analysis['total_samples']:,}
Valid Samples: {analysis['valid_samples']:,}
NaN Values: {analysis['nan_count']:,} ({analysis['nan_ratio']:.1%})

Distribution:
  • Mean: {analysis['mean']:.4f}
  • Median: {analysis['median']:.4f}
  • Std Dev: {analysis['std']:.4f}
  • Min: {analysis['min']:.4f}
  • Max: {analysis['max']:.4f}
  • Q25: {analysis['q25']:.4f}
  • Q75: {analysis['q75']:.4f}

Sign Analysis:
  • Positive: {analysis['positive_count']:,} ({analysis['positive_ratio']:.1%})
  • Negative: {analysis['negative_count']:,} ({analysis['negative_ratio']:.1%})
  • Zero: {analysis['zero_count']:,} ({analysis['zero_ratio']:.1%})
        """
        
        if analysis['positive_count'] > 0:
            stats_text += f"\n  • Pos Mean: {analysis['positive_mean']:.4f}"
            stats_text += f"\n  • Pos Std: {analysis['positive_std']:.4f}"
        
        if analysis['negative_count'] > 0:
            stats_text += f"\n  • Neg Mean: {analysis['negative_mean']:.4f}"
            stats_text += f"\n  • Neg Std: {analysis['negative_std']:.4f}"
        
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def plot_target_correlation_analysis(self, split: str = "train"):
        """Analyze correlation between move and value targets."""
        data = self.train_data if split == "train" else self.test_data
        move_targets = data.get('move_targets', np.array([]))
        value_targets = data.get('value_targets', np.array([]))
        
        if len(move_targets) == 0 or len(value_targets) == 0:
            print(f"Missing target data for correlation analysis in {split} split")
            return
        
        # Filter valid samples (both targets present and value not NaN)
        valid_mask = (move_targets != -100) & (~np.isnan(value_targets))
        
        if not valid_mask.any():
            print(f"No valid samples for correlation analysis in {split} split")
            return
        
        valid_moves = move_targets[valid_mask]
        valid_values = value_targets[valid_mask]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Move-Value Target Correlation Analysis ({split.title()} Set)', fontsize=16)
        
        # 1. Move frequency vs average value
        move_counter = Counter(valid_moves)
        move_ids, move_counts = zip(*move_counter.most_common(50))  # Top 50 moves
        
        # Calculate average value for each move
        move_avg_values = []
        for move_id in move_ids:
            move_mask = valid_moves == move_id
            avg_value = np.mean(valid_values[move_mask])
            move_avg_values.append(avg_value)
        
        # Color bars by average value
        bars = axes[0, 0].bar(range(len(move_ids)), move_counts, 
                             color=plt.cm.RdYlBu_r([0.5 + 0.5 * v / max(abs(min(move_avg_values)), abs(max(move_avg_values))) 
                                                     for v in move_avg_values]))
        axes[0, 0].set_title('Top 50 Moves: Frequency vs Average Value')
        axes[0, 0].set_xlabel('Move Rank')
        axes[0, 0].set_ylabel('Frequency')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, 
                                   norm=plt.Normalize(vmin=min(move_avg_values), vmax=max(move_avg_values)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[0, 0])
        cbar.set_label('Average Value')
        
        # 2. Value distribution by move popularity
        # Group moves by frequency quartiles
        move_frequencies = np.array(list(move_counter.values()))
        q25, q50, q75 = np.percentile(move_frequencies, [25, 50, 75])
        
        rare_moves = [mid for mid, count in move_counter.items() if count <= q25]
        common_moves = [mid for mid, count in move_counter.items() if q25 < count <= q75]
        frequent_moves = [mid for mid, count in move_counter.items() if count > q75]
        
        rare_values = valid_values[np.isin(valid_moves, rare_moves)]
        common_values = valid_values[np.isin(valid_moves, common_moves)]
        frequent_values = valid_values[np.isin(valid_moves, frequent_moves)]
        
        axes[0, 1].hist([rare_values, common_values, frequent_values], 
                       bins=30, alpha=0.7, 
                       label=['Rare Moves (Q1)', 'Common Moves (Q2-Q3)', 'Frequent Moves (Q4)'],
                       color=['lightcoral', 'lightyellow', 'lightgreen'],
                       edgecolor='black')
        axes[0, 1].set_title('Value Distribution by Move Frequency')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Scatter plot: Move ID vs Value
        sample_size = min(5000, len(valid_moves))  # Sample for performance
        sample_indices = np.random.choice(len(valid_moves), sample_size, replace=False)
        sample_moves = valid_moves[sample_indices]
        sample_values = valid_values[sample_indices]
        
        axes[1, 0].scatter(sample_moves, sample_values, alpha=0.5, s=1, color='blue')
        axes[1, 0].set_title(f'Move ID vs Value Scatter (Sample of {sample_size:,})')
        axes[1, 0].set_xlabel('Move ID')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Statistics comparison table
        axes[1, 1].axis('off')
        
        stats_text = f"""Target Correlation Statistics:

Total Valid Samples: {len(valid_moves):,}
Unique Moves: {len(move_counter):,}

Move Frequency Analysis:
  • Rarest moves (Q1): {len(rare_moves):,} moves, ≤{q25} occurrences
  • Common moves (Q2-Q3): {len(common_moves):,} moves, {q25+1}-{q75} occurrences  
  • Most frequent (Q4): {len(frequent_moves):,} moves, >{q75} occurrences

Value Statistics by Move Frequency:
Rare Moves:
  • Count: {len(rare_values):,}
  • Mean Value: {np.mean(rare_values):.4f}
  • Std Value: {np.std(rare_values):.4f}

Common Moves:
  • Count: {len(common_values):,}
  • Mean Value: {np.mean(common_values):.4f}
  • Std Value: {np.std(common_values):.4f}

Frequent Moves:
  • Count: {len(frequent_values):,}
  • Mean Value: {np.mean(frequent_values):.4f}
  • Std Value: {np.std(frequent_values):.4f}
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
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
        
        print(f"\nSample Data ({split.title()} Set):")
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
        print("\nDataset Validation:")
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
                    print(f"  [OK] {file_type}: {array.shape} {array.dtype}")
                    
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
                    print(f"  [MISSING] {file_type}")
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
            print(f"\n[WARNING] Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n[OK] Dataset validation passed!")
    
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
                print(f"\n{split.title()} Split Analysis:")
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
                
                # Value target analysis
                value_analysis = self.analyze_value_targets(split)
                if value_analysis and not value_analysis.get('all_nan', False):
                    print(f"  Value targets: {value_analysis['valid_samples']:,} valid ({value_analysis['nan_ratio']:.1%} NaN)")
                    print(f"  Value range: [{value_analysis['min']:.3f}, {value_analysis['max']:.3f}]")
                    print(f"  Value distribution: {value_analysis['positive_ratio']:.1%} pos, {value_analysis['zero_ratio']:.1%} zero, {value_analysis['negative_ratio']:.1%} neg")
                elif value_analysis:
                    print(f"  Value targets: All NaN values")
                else:
                    print(f"  Value targets: Not found")
        
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
                
                # New: value target plots
                if f'value_targets' in (visualizer.train_data if split == "train" else visualizer.test_data):
                    visualizer.plot_value_target_distribution(split)
                    visualizer.plot_target_correlation_analysis(split)


if __name__ == "__main__":
    main()