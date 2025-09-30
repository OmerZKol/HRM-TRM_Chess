#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
import gzip
import glob
import random
import multiprocessing as mp
from typing import Tuple, List, Optional
import argparse
import yaml
import os
from simple_chess_nn import ChessLoss, SimpleChessNet
from model.ChessNNet import ChessNNet
from transformer_chess_nn import TransformerChessNet
from torch.utils.tensorboard import SummaryWriter

# Training record version constants (from chunkparser.py)
V6_VERSION = struct.pack('i', 6)
V5_VERSION = struct.pack('i', 5)
V4_VERSION = struct.pack('i', 4)
V3_VERSION = struct.pack('i', 3)
CLASSICAL_INPUT = struct.pack('i', 1)

V6_STRUCT_STRING = '4si7432s832sBBBBBBBbfffffffffffffffIHH4H'
V5_STRUCT_STRING = '4si7432s832sBBBBBBBbfffffff'
V4_STRUCT_STRING = '4s7432s832sBBBBBBBbffff'
V3_STRUCT_STRING = '4s7432s832sBBBBBBBb'

def reverse_expand_bits(plane: int) -> bytes:
    """Convert single byte to expanded bit representation"""
    return np.unpackbits(np.array([plane], dtype=np.uint8))[::-1].astype(np.float32).tobytes()

class ChessDataset(Dataset):
    """PyTorch Dataset for chess training data"""
    
    def __init__(self, chunk_files: List[str], sample_rate: int = 1, 
                 expected_input_format: int = None, shuffle_size: int = 8192):
        self.chunk_files = chunk_files
        self.sample_rate = sample_rate
        self.expected_input_format = expected_input_format
        self.shuffle_size = shuffle_size
        
        # Initialize struct parsers
        self.v6_struct = struct.Struct(V6_STRUCT_STRING)
        self.v5_struct = struct.Struct(V5_STRUCT_STRING)
        self.v4_struct = struct.Struct(V4_STRUCT_STRING)
        self.v3_struct = struct.Struct(V3_STRUCT_STRING)
        
        # Pre-computed flat planes for efficiency
        self.flat_planes = {
            0: (64 * 4) * b'\x00',  # All zeros plane
            1: (64 * 4) * b'\x01'   # All ones plane  
        }
        
        # Load all training records
        self.records = self._load_all_records()
        
    def _load_all_records(self) -> List[bytes]:
        """Load and parse all training records from chunk files"""
        records = []
        
        for chunk_file in self.chunk_files:
            try:
                if chunk_file.endswith('.gz'):
                    with gzip.open(chunk_file, 'rb') as f:
                        chunk_data = f.read()
                else:
                    with open(chunk_file, 'rb') as f:
                        chunk_data = f.read()
                
                records.extend(self._sample_records(chunk_data))
                
            except Exception as e:
                print(f"Error loading {chunk_file}: {e}")
                
        random.shuffle(records)
        return records
    
    def _sample_records(self, chunkdata: bytes) -> List[bytes]:
        """Sample records from chunk data with downsampling"""
        version = chunkdata[0:4]
        
        if version == V6_VERSION:
            record_size = self.v6_struct.size
        elif version == V5_VERSION:
            record_size = self.v5_struct.size
        elif version == V4_VERSION:
            record_size = self.v4_struct.size
        elif version == V3_VERSION:
            record_size = self.v3_struct.size
        else:
            return []
            
        records = []
        
        for i in range(0, len(chunkdata), record_size):
            if self.sample_rate > 1:
                if random.randint(0, self.sample_rate - 1) != 0:
                    continue
                    
            record = chunkdata[i:i + record_size]
            
            # Pad older versions to V6 format
            if version == V3_VERSION:
                record += 16 * b'\x00'  # Add fake root_q, best_q, root_d, best_d
            if version == V3_VERSION or version == V4_VERSION:
                record += 12 * b'\x00'  # Add fake root_m, best_m, plies_left
                record = record[:4] + CLASSICAL_INPUT + record[4:]  # Insert input format
            if version in [V3_VERSION, V4_VERSION, V5_VERSION]:
                record += 48 * b'\x00'  # Add fake result_q, result_d etc
                
            records.append(record)
            
        return records
    
    def _create_dummy_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a dummy sample when data parsing fails"""
        planes = torch.zeros(112, 8, 8, dtype=torch.float32)
        policy = torch.zeros(1858, dtype=torch.float32)
        policy[0] = 1.0  # Set first move as 100% probability
        value = torch.tensor([0.33, 0.34, 0.33], dtype=torch.float32)  # Balanced WDL
        best_q = torch.tensor([0.33, 0.34, 0.33], dtype=torch.float32)
        moves_left = torch.tensor([40.0], dtype=torch.float32)
        return planes, policy, value, best_q, moves_left
    
    def _convert_v6_to_tuple(self, content: bytes) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert V6 binary record to tensors (adapted from chunkparser.py)"""
        
        # Unpack the V6 content
        (ver, input_format, probs, planes, us_ooo, us_oo, them_ooo, them_oo,
         stm, rule50_count, invariance_info, dep_result, root_q, best_q,
         root_d, best_d, root_m, best_m, plies_left, result_q, result_d,
         played_q, played_d, played_m, orig_q, orig_d, orig_m, visits,
         played_idx, best_idx, reserved1, reserved2, reserved3, reserved4) = self.v6_struct.unpack(content)
        
        # Handle plies_left fallback
        if plies_left == 0:
            plies_left = invariance_info
        # Auto-detect input format instead of asserting
        if self.expected_input_format is None:
            self.expected_input_format = input_format
        elif input_format != self.expected_input_format:
            print(f"Warning: Expected input format {self.expected_input_format}, got {input_format}")
            self.expected_input_format = input_format  # Update to match data
        
        # Unpack bit planes and cast to float32
        planes_array = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32)
        
        # Rule50 plane
        rule50_divisor = 100.0 if input_format > 3 else 99.0
        rule50_plane = struct.pack('f', rule50_count / rule50_divisor) * 64
        
        # Handle different input formats for middle planes
        if input_format == 1:
            # Classic input format - simpler plane structure using indices
            middle_planes = (self.flat_planes[min(1, us_ooo)] + 
                           self.flat_planes[min(1, us_oo)] + 
                           self.flat_planes[min(1, them_ooo)] + 
                           self.flat_planes[min(1, them_oo)] + 
                           self.flat_planes[min(1, stm)])
        elif input_format in [3, 4, 132, 5, 133]:
            them_ooo_bytes = reverse_expand_bits(them_ooo)
            us_ooo_bytes = reverse_expand_bits(us_ooo)
            them_oo_bytes = reverse_expand_bits(them_oo)
            us_oo_bytes = reverse_expand_bits(us_oo)
            enpassant_bytes = reverse_expand_bits(stm)
            
            middle_planes = (us_ooo_bytes + (6*8*4) * b'\x00' + them_ooo_bytes +
                           us_oo_bytes + (6*8*4) * b'\x00' + them_oo_bytes +
                           self.flat_planes[0] + self.flat_planes[0] +
                           (7*8*4) * b'\x00' + enpassant_bytes)
        else:
            # Handle other input formats as needed
            middle_planes = self.flat_planes[0] * 8
        
        # Edge detection plane
        aux_plus_6_plane = self.flat_planes[0]
        if (input_format in [132, 133]) and invariance_info >= 128:
            aux_plus_6_plane = self.flat_planes[1]
            
        # Concatenate all planes
        all_planes = (planes_array.tobytes() + middle_planes + 
                     rule50_plane + aux_plus_6_plane + self.flat_planes[1])
        
        # Calculate expected length based on input format
        if input_format == 1:
            # 104 planes from unpacked bits + 8 auxiliary planes = 112 planes total
            expected_len = 112 * 8 * 8 * 4  # 112 planes × 64 squares × 4 bytes per float
        else:
            expected_len = ((8 * 13 * 1 + 8 * 1 * 1) * 8 * 8 * 4)
        
        actual_len = len(all_planes)
        if actual_len != expected_len:
            print(f"Plane length mismatch: expected {expected_len}, got {actual_len}")
            print(f"Input format: {input_format}")
            print(f"planes_array: {len(planes_array.tobytes())} (should be {104*8*8*4})")
            print(f"middle_planes: {len(middle_planes)} (should be {5*8*8*4} for format 1)")
            print(f"rule50_plane: {len(rule50_plane)}")
            print(f"aux_plus_6_plane: {len(aux_plus_6_plane)}")
            print(f"flat_planes[1]: {len(self.flat_planes[1])}")
            # Skip this record instead of crashing
            return self._create_dummy_sample()
        
        # assert len(all_planes) == ((8 * 13 * 1 + 8 * 1 * 1) * 8 * 8 * 4)
        
        # Convert to tensor and reshape to (112, 8, 8)
        planes_tensor = torch.frombuffer(all_planes, dtype=torch.float32).view(112, 8, 8)
        
        # Policy probabilities
        probs_array = np.frombuffer(probs, dtype=np.float32)
        policy_tensor = torch.from_numpy(probs_array)
        
        # Value targets (WDL format)
        if ver == V6_VERSION:
            value_tensor = torch.tensor([
                0.5 * (1.0 - result_d + result_q),  # Win probability
                result_d,                            # Draw probability  
                0.5 * (1.0 - result_d - result_q)   # Loss probability
            ], dtype=torch.float32)
        else:
            dep_result = float(dep_result)
            value_tensor = torch.tensor([
                float(dep_result == 1.0),   # Win
                float(dep_result == 0.0),   # Draw
                float(dep_result == -1.0)   # Loss
            ], dtype=torch.float32)
        
        # Best Q value (also in WDL format)
        best_q_w = 0.5 * (1.0 - best_d + best_q)
        best_q_l = 0.5 * (1.0 - best_d - best_q)
        best_q_tensor = torch.tensor([best_q_w, best_d, best_q_l], dtype=torch.float32)
        
        # Moves left
        moves_left_tensor = torch.tensor([plies_left], dtype=torch.float32)
        
        return planes_tensor, policy_tensor, value_tensor, best_q_tensor, moves_left_tensor
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        return self._convert_v6_to_tuple(record)

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: ChessLoss,
            device: torch.device) -> dict:
    """Evaluate model on validation set"""
    model.eval()
    total_losses = {'policy_loss': 0, 'value_loss': 0, 'moves_left_loss': 0, 
                   'reg_loss': 0, 'total_loss': 0, "move_accuracy": 0, 'q_loss': 0}
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (planes, policy_target, value_target, best_q_target, ml_target) in enumerate(dataloader):
            planes = planes.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            ml_target = ml_target.to(device)

            # for HRM model, compute output with mixed precision for FlashAttention compatibility
            if(criterion.model_type == "hrm"):
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    model_output = model(planes)
            else:
                model_output = model(planes)
            
            policy_output, value_output, moves_left_output, q_output = model_output

            # Calculate loss
            total_loss, loss_dict = criterion(policy_target, policy_output, 
                                  value_target, value_output,
                                  ml_target, moves_left_output,
                                  q_output, model)
            
            # Accumulate losses
            total_losses['policy_loss'] += loss_dict['policy_loss']
            total_losses['value_loss'] += loss_dict['value_loss']
            total_losses['moves_left_loss'] += loss_dict['moves_left_loss']
            total_losses['reg_loss'] += loss_dict['reg_loss']
            total_losses['q_loss'] += loss_dict['q_loss']
            total_losses['total_loss'] += loss_dict['total_loss']
            total_losses['move_accuracy'] += loss_dict['policy_accuracy']

            num_batches += 1

    # Average losses and accuracy
    for key in total_losses:
        total_losses[key] /= num_batches

    return total_losses

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion, 
                optimizer: torch.optim.Optimizer, device: torch.device) -> dict:
    """Train for one epoch"""
    model.train()
    
    total_losses = {'policy_loss': 0, 'value_loss': 0, 'moves_left_loss': 0, 
                   'reg_loss': 0, 'q_halt_loss': 0, 'q_continue_loss': 0, 'q_loss': 0,
                    'total_loss': 0, 'move_accuracy': 0}
    num_batches = 0
    
    for batch_idx, (planes, policy_target, value_target, best_q_target, ml_target) in enumerate(dataloader):
        planes = planes.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)
        ml_target = ml_target.to(device)
        
        optimizer.zero_grad()
        
        # # Forward pass - assuming model returns (policy_logits, value_logits, moves_left)
        
        # compute output with mixed precision for FlashAttention compatibility
        if(criterion.model_type == "hrm"):
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                model_output = model(planes)
        else:
            model_output = model(planes)
        policy_output, value_output, moves_left_output, q_output = model_output
        
        # Calculate loss
        total_loss, loss_dict = criterion(policy_target, policy_output, 
                                  value_target, value_output,
                                  ml_target, moves_left_output,
                                  q_output, model)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_losses['policy_loss'] += loss_dict['policy_loss']
        total_losses['value_loss'] += loss_dict['value_loss']
        total_losses['moves_left_loss'] += loss_dict['moves_left_loss']
        total_losses['q_halt_loss'] += loss_dict['q_halt_loss']
        total_losses['q_continue_loss'] += loss_dict['q_continue_loss']
        total_losses['q_loss'] += loss_dict['q_loss']
        total_losses['reg_loss'] += loss_dict['reg_loss']
        total_losses['total_loss'] += loss_dict['total_loss']
        total_losses['move_accuracy'] += loss_dict['policy_accuracy']
        num_batches += 1
    
    # Average losses and accuracy
    for key in total_losses:
        total_losses[key] /= num_batches
        
    return total_losses

def load_model(args, config, device):
    """Load model based on command line arguments"""
    
    # If model path is provided, try to load it
    if args.model_path:
        if args.model_path.endswith('.pth'):
            print(f"Loading PyTorch model from {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=device)

            if config.get("model_type") == 'simple':
                model = SimpleChessNet()
                # Handle both direct state_dict and nested checkpoint format
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
            else:
                model = ChessNNet(config, (8,8))
            return model.to(device)
    if(config.get("model_type") == "simple"):
        return SimpleChessNet().to(device)
    if(config.get("model_type") == "hrm"):
        return ChessNNet(config, (8,8)).to(device)
    if(config.get("model_type") == "transformer"):
        return TransformerChessNet((8,8), 1858).to(device)

def save_model(model, optimizer, scheduler, config, args, epoch, losses):
    """Save model with training info"""
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = f"model_checkpoints/chess_model_epoch_{epoch}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model state and training info
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'losses': losses,
        'model_type': config.get("model_type"),
        'model_class': model.__class__.__name__
    }
    
    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='PyTorch Chess Training')
    parser.add_argument('--data-path', type=str, required=False, default="data/training-run3--20210605-0521",
                       help='Path to training data chunks')
    parser.add_argument('--config', type=str,
                       help='YAML config file path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--model-path', type=str,
                       help='Path to existing PyTorch model (.pth)')
    parser.add_argument('--save-path', type=str,
                       help='Path to save trained model (.pth)') 
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model
    model = load_model(args, config, device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    print(args.data_path)
    # Get chunk files
    chunk_files = glob.glob(os.path.join(args.data_path, "*.gz"))
    print(f'Found {len(chunk_files)} chunk files in {args.data_path}')
    
    if not chunk_files:
        print("No chunk files found! Use --test flag to run with dummy data.")
        return

    chunk_files = chunk_files[:20]

    train_split = int(0.9 * len(chunk_files)) # 90% for training, 10% for validation

    train_dataset = ChessDataset(chunk_files[:train_split], sample_rate=config.get('sample_rate', 0))  # Use higher sampling for speed
    valid_dataset = ChessDataset(chunk_files[train_split:], sample_rate=config.get('sample_rate', 0))
    train_dataloader = DataLoader(train_dataset, batch_size=config.get('batch_size', 64),
                            shuffle=True, num_workers=0, pin_memory=False)
    test_dataloader = DataLoader(valid_dataset, batch_size=config.get('batch_size', 64), 
                            shuffle=False, num_workers=0, pin_memory=False)

    print(f'Training dataset size: {len(train_dataset)}')

    # Create loss function and optimizer
    criterion = ChessLoss(
        policy_weight=config.get('policy_loss_weight', 1.0),
        value_weight=config.get('value_loss_weight', 1.0), 
        moves_left_weight=config.get('moves_left_loss_weight', 0.01),
        config=config
    )
    for param in model.parameters():
        param.requires_grad = True
        # print(param.shape, param.dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.0001))
    print(f'Optimizer: Adam with lr={config.get("lr", 0.0001)}')
    # Add learning rate scheduler - cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=config.get('lr', 0.0001)/100
    )
    
    # Initialize TensorBoard writer
    log_dir = f"runs/chess_training_{config.get('name')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f'TensorBoard logs will be saved to: {log_dir}')
    
    # Initialize parameter tracking
    param_snapshot_dir = "parameter_snapshots"
    os.makedirs(param_snapshot_dir, exist_ok=True)
    print(f'Parameter snapshots will be saved to: {param_snapshot_dir}')

    # Training loop
    for epoch in range(config.get('epochs')):
        print(f'\nEpoch {epoch+1}/{config.get('epochs')}')
        
        losses = train_epoch(model, train_dataloader, criterion, optimizer, device)
        
        # Step the learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Average Losses (LR: {current_lr:.2e}):')
        for key, value in losses.items():
            print(f'  {key}: {value:.6f}')
        
        # Log training metrics to TensorBoard
        for key, value in losses.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        if(epoch % 2 == 0):
            eval_losses = evaluate(model, test_dataloader, criterion, device)
            print(f'Validation Losses:')
            for key, value in eval_losses.items():
                print(f'  {key}: {value:.6f}')
            
            # Log validation metrics to TensorBoard
            for key, value in eval_losses.items():
                writer.add_scalar(f'Validation/{key}', value, epoch)
            
            # Save model checkpoint
            save_model(model, optimizer, scheduler, config, args, epoch, losses)

    # Close TensorBoard writer
    writer.close()
    print(f"Training completed. View logs with: tensorboard --logdir {log_dir}")
    print(f"Parameter snapshots saved to: {param_snapshot_dir}")
    print(f"To visualize parameter evolution, run: python visualize_parameters.py --snapshot-dir {param_snapshot_dir}")

if __name__ == '__main__':
    main()