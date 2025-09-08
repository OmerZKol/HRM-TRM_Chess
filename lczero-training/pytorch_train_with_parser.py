#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import glob
import sys
import os
import argparse
import yaml
from typing import Tuple, Iterator

# Add the tf directory to Python path to import chunkparser
sys.path.append(os.path.join(os.path.dirname(__file__), 'tf'))

try:
    from chunkparser import ChunkParser
except ImportError as e:
    print(f"Error importing chunkparser: {e}")
    print("Make sure you're running from the lczero-training directory")
    sys.exit(1)

class LczeroIterableDataset(IterableDataset):
    """PyTorch IterableDataset using Lczero's ChunkParser"""
    
    def __init__(self, chunk_files, expected_input_format=1, shuffle_size=8192, 
                 sample_rate=1, batch_size=256, workers=4):
        self.chunk_files = chunk_files
        self.expected_input_format = expected_input_format
        self.shuffle_size = shuffle_size
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.workers = workers
        self.parser = None
        
    def _create_parser(self):
        """Create ChunkParser instance"""
        if self.parser is None:
            self.parser = ChunkParser(
                chunks=self.chunk_files,
                expected_input_format=self.expected_input_format,
                shuffle_size=self.shuffle_size,
                sample=self.sample_rate,
                batch_size=self.batch_size,
                workers=self.workers
            )
    
    def _convert_to_tensors(self, batch_data):
        """Convert ChunkParser output to PyTorch tensors"""
        # batch_data is a tuple of (planes_bytes, probs_bytes, winner_bytes, best_q_bytes, plies_left_bytes)
        # where each element is concatenated bytes from the entire batch
        planes_bytes, probs_bytes, winner_bytes, best_q_bytes, plies_left_bytes = batch_data
        
        # Calculate sizes
        planes_size = 112 * 8 * 8 * 4  # 112 planes * 64 squares * 4 bytes per float32
        probs_size = 1858 * 4  # 1858 moves * 4 bytes per float32
        winner_size = 3 * 4  # 3 WDL values * 4 bytes per float32
        best_q_size = 3 * 4  # 3 WDL values * 4 bytes per float32
        plies_left_size = 1 * 4  # 1 value * 4 bytes per float32
        
        # Determine batch size
        batch_size = len(planes_bytes) // planes_size
        
        # Convert planes
        planes_array = np.frombuffer(planes_bytes, dtype=np.float32)
        planes_tensor = torch.from_numpy(planes_array.copy()).view(batch_size, 112, 8, 8)
        
        # Convert policy probabilities
        probs_array = np.frombuffer(probs_bytes, dtype=np.float32)
        policy_tensor = torch.from_numpy(probs_array.copy()).view(batch_size, 1858)
        
        # Convert value (WDL)
        winner_array = np.frombuffer(winner_bytes, dtype=np.float32)
        value_tensor = torch.from_numpy(winner_array.copy()).view(batch_size, 3)
        
        # Convert best Q (WDL)
        best_q_array = np.frombuffer(best_q_bytes, dtype=np.float32)
        best_q_tensor = torch.from_numpy(best_q_array.copy()).view(batch_size, 3)
        
        # Convert moves left
        plies_left_array = np.frombuffer(plies_left_bytes, dtype=np.float32)
        moves_left_tensor = torch.from_numpy(plies_left_array.copy()).view(batch_size, 1)
        
        return (planes_tensor, policy_tensor, value_tensor, best_q_tensor, moves_left_tensor)
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over batches of training data"""
        self._create_parser()
        
        try:
            for batch_data in self.parser.parse():
                yield self._convert_to_tensors(batch_data)
                
        except Exception as e:
            print(f"Error in data iteration: {e}")
            if self.parser:
                self.parser.shutdown()
            raise
    
    def shutdown(self):
        """Shutdown the chunk parser"""
        if self.parser:
            self.parser.shutdown()
            self.parser = None

class SimpleChessNet(nn.Module):
    """Simple test architecture for chess training"""
    
    def __init__(self, input_channels=112, hidden_dim=256, policy_size=1858):
        super().__init__()
        
        # Convolutional layers for spatial features
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_size)
        
        # Value head  
        self.value_fc1 = nn.Linear(256, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, 3)  # WDL output
        
        # Moves left head
        self.moves_left_fc1 = nn.Linear(256, hidden_dim // 2)
        self.moves_left_fc2 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        # x shape: (batch_size, 112, 8, 8)
        
        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))
        
        # Policy head - keep spatial dimensions
        policy_features = F.relu(self.policy_conv(x))
        policy_flat = policy_features.view(policy_features.size(0), -1)
        policy_logits = self.policy_fc(policy_flat)
        
        # Value and moves left heads - use global pooling
        global_features = self.global_pool(x).view(x.size(0), -1)
        
        # Value head (WDL)
        value_hidden = F.relu(self.value_fc1(global_features))
        value_logits = self.value_fc2(value_hidden)
        
        # Moves left head
        moves_hidden = F.relu(self.moves_left_fc1(global_features))
        moves_left = self.moves_left_fc2(moves_hidden)
        
        return policy_logits, value_logits, moves_left

class ChessLoss(nn.Module):
    """Combined loss function for chess training"""
    
    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0, 
                 moves_left_weight: float = 0.01, reg_weight: float = 1e-4):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.moves_left_weight = moves_left_weight
        self.reg_weight = reg_weight
        
    def policy_loss(self, policy_target: torch.Tensor, policy_output: torch.Tensor) -> torch.Tensor:
        """Policy cross-entropy loss with proper normalization"""
        # The policy target from ChunkParser appears to be raw probabilities (not one-hot)
        # Convert to probabilities and normalize
        policy_target_normalized = F.softmax(policy_target, dim=1)
        
        # Use KL divergence loss since we have probability distributions
        policy_output_log_probs = F.log_softmax(policy_output, dim=1)
        return F.kl_div(policy_output_log_probs, policy_target_normalized, reduction='batchmean')
    
    def value_loss(self, value_target: torch.Tensor, value_output: torch.Tensor) -> torch.Tensor:
        """WDL value cross-entropy loss"""
        return F.cross_entropy(value_output, value_target, reduction='mean')
    
    def moves_left_loss(self, ml_target: torch.Tensor, ml_output: torch.Tensor) -> torch.Tensor:
        """Huber loss for moves left prediction"""
        scale = 20.0
        ml_target_scaled = ml_target / scale
        ml_output_scaled = ml_output / scale
        
        return F.huber_loss(ml_output_scaled, ml_target_scaled, delta=10.0/scale)
    
    def forward(self, policy_target: torch.Tensor, policy_output: torch.Tensor,
                value_target: torch.Tensor, value_output: torch.Tensor,
                ml_target: torch.Tensor, ml_output: torch.Tensor,
                model: nn.Module) -> Tuple[torch.Tensor, dict]:
        """Combined loss calculation"""
        
        # Individual losses
        p_loss = self.policy_loss(policy_target, policy_output)
        v_loss = self.value_loss(value_target, value_output)
        ml_loss = self.moves_left_loss(ml_target, ml_output)
        
        # L2 regularization
        reg_loss = sum(p.pow(2.0).sum() for p in model.parameters()) * self.reg_weight
        
        # Combined loss
        total_loss = (self.policy_weight * p_loss + 
                     self.value_weight * v_loss + 
                     self.moves_left_weight * ml_loss + 
                     reg_loss)
        
        # Return loss components for logging
        loss_dict = {
            'policy_loss': p_loss.item(),
            'value_loss': v_loss.item(), 
            'moves_left_loss': ml_loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: ChessLoss, 
                optimizer: torch.optim.Optimizer, device: torch.device, max_batches: int = None) -> dict:
    """Train for one epoch with optional batch limit"""
    model.train()
    
    total_losses = {'policy_loss': 0, 'value_loss': 0, 'moves_left_loss': 0, 
                   'reg_loss': 0, 'total_loss': 0}
    num_batches = 0
    
    try:
        for batch_idx, (planes, policy_target, value_target, best_q_target, ml_target) in enumerate(dataloader):
            planes = planes.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            ml_target = ml_target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            policy_output, value_output, ml_output = model(planes)
            
            # Calculate loss
            loss, loss_dict = criterion(policy_target, policy_output, 
                                      value_target, value_output,
                                      ml_target, ml_output, model)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}, Total Loss: {loss_dict["total_loss"]:.4f}')
            
            # Break early if max_batches specified
            if max_batches and batch_idx >= max_batches:
                break
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    # Average losses
    if num_batches > 0:
        for key in total_losses:
            total_losses[key] /= num_batches
    
    return total_losses

def main():
    parser = argparse.ArgumentParser(description='PyTorch Chess Training with ChunkParser')
    parser.add_argument('--data-path', type=str, 
                       default='/home/omerz/projects/ChessTypeBeat/lczero-training/data/training-run3--20210605-0521',
                       help='Path to training data chunks')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--config', type=str,
                       help='YAML config file path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--max-files', type=int, default=100,
                       help='Maximum number of chunk files to use')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum number of batches per epoch (for testing)')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of ChunkParser worker processes')
    parser.add_argument('--save-model', type=str, 
                       help='Path to save the trained model')
    parser.add_argument('--shuffle-size', type=int, default=8192,
                       help='Size of shuffle buffer')
    parser.add_argument('--sample-rate', type=int, default=32,
                       help='Sample rate for training data (1=all data, 32=every 32nd sample)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get chunk files
    chunk_files = glob.glob(os.path.join(args.data_path, "*.gz"))
    
    if not chunk_files:
        print(f"No chunk files found in {args.data_path}")
        return
    
    # Limit number of files for faster testing
    if len(chunk_files) > args.max_files:
        print(f"Using first {args.max_files} files out of {len(chunk_files)}")
        chunk_files = chunk_files[:args.max_files]
    
    print(f'Found {len(chunk_files)} chunk files')
    
    # Initialize model
    model = SimpleChessNet().to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create dataset and dataloader
    dataset = LczeroIterableDataset(
        chunk_files=chunk_files,
        expected_input_format=1,  # Classic format from our data inspection
        shuffle_size=args.shuffle_size,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        workers=args.workers
    )
    
    # Note: batch_size=None for IterableDataset since batching is handled by ChunkParser
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    
    # Create loss function and optimizer
    criterion = ChessLoss(
        policy_weight=config.get('policy_loss_weight', 1.0),
        value_weight=config.get('value_loss_weight', 1.0), 
        moves_left_weight=config.get('moves_left_loss_weight', 0.01)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    try:
        for epoch in range(args.epochs):
            print(f'\nEpoch {epoch+1}/{args.epochs}')
            
            losses = train_epoch(model, dataloader, criterion, optimizer, device, 
                               max_batches=args.max_batches)
            
            print(f'Average Losses:')
            for key, value in losses.items():
                print(f'  {key}: {value:.6f}')
            
            # Save model checkpoint
            if args.save_model and (epoch + 1) % 5 == 0:
                checkpoint_path = f"{args.save_model}_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses
                }, checkpoint_path)
                print(f'Model saved to {checkpoint_path}')
        
        # Save final model
        if args.save_model:
            final_path = f"{args.save_model}_final.pth"
            torch.save({
                'epoch': args.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses
            }, final_path)
            print(f'Final model saved to {final_path}')
    
    finally:
        # Always shutdown the dataset to clean up workers
        print("Shutting down dataset...")
        dataset.shutdown()

if __name__ == '__main__':
    main()