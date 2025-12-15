#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import glob
import argparse
import yaml
import os
from chess_loss import ChessLoss
from model.SimpleChessNet import SimpleChessNet
from model.ChessNNet import ChessNNet
from model.ChessTRMNet import ChessTRMNet
from model.ChessTRMBaselineNet import ChessTRMBaselineNet
from transformer_chess_nn import TransformerChessNet
from torch.utils.tensorboard import SummaryWriter
from chess_dataset import ChessDataset

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: ChessLoss,
            device: torch.device) -> dict:
    """Evaluate model on validation set"""
    model.eval()
    total_losses = {'policy_loss': 0, 'value_loss': 0, 'moves_left_loss': 0,
                   'reg_loss': 0, 'total_loss': 0, "move_accuracy": 0, 'q_loss': 0}
    num_batches = 0
    total_recursion_steps = 0
    num_recursion_samples = 0

    with torch.no_grad():
        for batch_idx, (planes, policy_target, value_target, best_q_target, ml_target) in enumerate(dataloader):
            planes = planes.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            ml_target = ml_target.to(device)

            # NaN detection: Check validation input data
            if torch.isnan(planes).any():
                print(f"[NaN Detection - Validation] Batch {batch_idx}: NaN in input planes")
            if torch.isnan(policy_target).any():
                print(f"[NaN Detection - Validation] Batch {batch_idx}: NaN in policy_target")
            if torch.isnan(value_target).any():
                print(f"[NaN Detection - Validation] Batch {batch_idx}: NaN in value_target")
            if torch.isnan(ml_target).any():
                print(f"[NaN Detection - Validation] Batch {batch_idx}: NaN in ml_target")

            # for HRM/TRM/TRM_baseline model, compute output with mixed precision for FlashAttention compatibility
            if(criterion.model_type == "hrm" or criterion.model_type == "trm" or criterion.model_type == "trm_baseline"):
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    model_output = model(planes)
            else:
                model_output = model(planes)

            policy_output, value_output, moves_left_output, q_output = model_output

            # NaN detection: Check validation model outputs
            if torch.isnan(policy_output).any():
                print(f"[NaN Detection - Validation] Batch {batch_idx}: NaN in policy_output")
                raise RuntimeError(f"NaN detected in validation policy_output at batch {batch_idx}. Stopping training.")
            if torch.isnan(value_output).any():
                print(f"[NaN Detection - Validation] Batch {batch_idx}: NaN in value_output")
                raise RuntimeError(f"NaN detected in validation value_output at batch {batch_idx}. Stopping training.")
            if torch.isnan(moves_left_output).any():
                print(f"[NaN Detection - Validation] Batch {batch_idx}: NaN in moves_left_output")
                raise RuntimeError(f"NaN detected in validation moves_left_output at batch {batch_idx}. Stopping training.")

            # Calculate loss
            total_loss, loss_dict = criterion(policy_target, policy_output,
                                  value_target, value_output,
                                  ml_target, moves_left_output,
                                  q_output, model)

            # NaN detection: Check validation loss
            if torch.isnan(total_loss):
                print(f"[NaN Detection - Validation] Batch {batch_idx}: NaN in total_loss")
                print(f"  Loss components: {loss_dict}")
                raise RuntimeError(f"NaN detected in validation total_loss at batch {batch_idx}. Stopping training.")

            # Track recursion steps if available (HRM/TRM models)
            if "recursion_steps" in q_output:
                recursion_steps = q_output["recursion_steps"]
                total_recursion_steps += recursion_steps.sum().item()
                num_recursion_samples += recursion_steps.numel()

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

    # Add average recursion steps if tracked
    if num_recursion_samples > 0:
        total_losses['avg_recursion_steps'] = total_recursion_steps / num_recursion_samples
    else:
        total_losses['avg_recursion_steps'] = 0.0

    return total_losses

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion,
                optimizer: torch.optim.Optimizer, device: torch.device,
                scaler: torch.cuda.amp.GradScaler = None,
                gradient_accumulation_steps: int = 1) -> dict:
    """Train for one epoch with optional mixed precision and gradient accumulation"""
    model.train()

    total_losses = {'policy_loss': 0, 'value_loss': 0, 'moves_left_loss': 0,
                   'reg_loss': 0, 'q_halt_loss': 0, 'q_continue_loss': 0, 'q_loss': 0,
                    'total_loss': 0, 'move_accuracy': 0}
    num_batches = 0
    total_recursion_steps = 0
    num_recursion_samples = 0

    for batch_idx, (planes, policy_target, value_target, best_q_target, ml_target) in enumerate(dataloader):
        planes = planes.to(device, non_blocking=True)
        policy_target = policy_target.to(device, non_blocking=True)
        value_target = value_target.to(device, non_blocking=True)
        ml_target = ml_target.to(device, non_blocking=True)

        # NaN detection: Check input data
        if torch.isnan(planes).any():
            print(f"[NaN Detection] Batch {batch_idx}: NaN detected in input planes")
        if torch.isnan(policy_target).any():
            print(f"[NaN Detection] Batch {batch_idx}: NaN detected in policy_target")
        if torch.isnan(value_target).any():
            print(f"[NaN Detection] Batch {batch_idx}: NaN detected in value_target")
        if torch.isnan(ml_target).any():
            print(f"[NaN Detection] Batch {batch_idx}: NaN detected in ml_target")

        # Forward pass with automatic mixed precision
        use_amp = scaler is not None
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                model_output = model(planes)
                policy_output, value_output, moves_left_output, q_output = model_output

                # NaN detection: Check model outputs
                if torch.isnan(policy_output).any():
                    print(f"[NaN Detection] Batch {batch_idx}: NaN detected in policy_output (after forward)")
                    raise RuntimeError(f"NaN detected in policy_output at batch {batch_idx}. Stopping training.")
                if torch.isnan(value_output).any():
                    print(f"[NaN Detection] Batch {batch_idx}: NaN detected in value_output (after forward)")
                    raise RuntimeError(f"NaN detected in value_output at batch {batch_idx}. Stopping training.")
                if torch.isnan(moves_left_output).any():
                    print(f"[NaN Detection] Batch {batch_idx}: NaN detected in moves_left_output (after forward)")
                    raise RuntimeError(f"NaN detected in moves_left_output at batch {batch_idx}. Stopping training.")

                # Calculate loss
                total_loss, loss_dict = criterion(policy_target, policy_output,
                                          value_target, value_output,
                                          ml_target, moves_left_output,
                                          q_output, model)
                # Scale loss for gradient accumulation
                total_loss = total_loss / gradient_accumulation_steps
        else:
            # compute output with mixed precision for FlashAttention compatibility (HRM/TRM/TRM_baseline only)
            if(criterion.model_type == "hrm" or criterion.model_type == "trm" or criterion.model_type == "trm_baseline"):
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    model_output = model(planes)
                    policy_output, value_output, moves_left_output, q_output = model_output

                    # NaN detection: Check model outputs
                    if torch.isnan(policy_output).any():
                        print(f"[NaN Detection] Batch {batch_idx}: NaN detected in policy_output (after forward)")
                        print(f"Policy output stats: min={policy_output.min().item():.4f}, max={policy_output.max().item():.4f}, mean={policy_output.mean().item():.4f}")
                        print(f"Number of NaN values: {torch.isnan(policy_output).sum().item()}")
                        print(f"Number of Inf values: {torch.isinf(policy_output).sum().item()}")
                        # Check individual positions in batch
                        for i in range(min(3, policy_output.shape[0])):
                            if torch.isnan(policy_output[i]).any():
                                print(f"  Batch item {i} has NaN. Stats: min={policy_output[i].min().item():.4f}, max={policy_output[i].max().item():.4f}")
                        raise RuntimeError(f"NaN detected in policy_output at batch {batch_idx}. Stopping training.")
                    if torch.isnan(value_output).any():
                        print(f"[NaN Detection] Batch {batch_idx}: NaN detected in value_output (after forward)")
                        raise RuntimeError(f"NaN detected in value_output at batch {batch_idx}. Stopping training.")
                    if torch.isnan(moves_left_output).any():
                        print(f"[NaN Detection] Batch {batch_idx}: NaN detected in moves_left_output (after forward)")
                        raise RuntimeError(f"NaN detected in moves_left_output at batch {batch_idx}. Stopping training.")

                    # Calculate loss inside autocast context
                    total_loss, loss_dict = criterion(policy_target, policy_output,
                                              value_target, value_output,
                                              ml_target, moves_left_output,
                                              q_output, model)
            else:
                model_output = model(planes)
                policy_output, value_output, moves_left_output, q_output = model_output

                # NaN detection: Check model outputs
                if torch.isnan(policy_output).any():
                    print(f"[NaN Detection] Batch {batch_idx}: NaN detected in policy_output (after forward)")
                    print(policy_output)
                    print(value_output)
                    print(moves_left_output)
                    raise RuntimeError(f"NaN detected in policy_output at batch {batch_idx}. Stopping training.")
                if torch.isnan(value_output).any():
                    print(f"[NaN Detection] Batch {batch_idx}: NaN detected in value_output (after forward)")
                    raise RuntimeError(f"NaN detected in value_output at batch {batch_idx}. Stopping training.")
                if torch.isnan(moves_left_output).any():
                    print(f"[NaN Detection] Batch {batch_idx}: NaN detected in moves_left_output (after forward)")
                    raise RuntimeError(f"NaN detected in moves_left_output at batch {batch_idx}. Stopping training.")

                # Calculate loss
                total_loss, loss_dict = criterion(policy_target, policy_output,
                                          value_target, value_output,
                                          ml_target, moves_left_output,
                                          q_output, model)
            # Scale loss for gradient accumulation
            total_loss = total_loss / gradient_accumulation_steps

        # NaN detection: Check loss
        if torch.isnan(total_loss):
            print(f"[NaN Detection] Batch {batch_idx}: NaN detected in total_loss")
            print(f"  Loss components: {loss_dict}")
            # Check individual loss components
            for key, value in loss_dict.items():
                if isinstance(value, float) and (value != value):  # NaN check for float
                    print(f"  - {key} is NaN")
            raise RuntimeError(f"NaN detected in total_loss at batch {batch_idx}. Stopping training.")

        # Backward pass
        if use_amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # NaN detection: Check gradients
        nan_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_grads.append(name)
        if nan_grads:
            print(f"[NaN Detection] Batch {batch_idx}: NaN detected in gradients:")
            for name in nan_grads[:5]:  # Show first 5 to avoid spam
                print(f"  - {name}")
            if len(nan_grads) > 5:
                print(f"  ... and {len(nan_grads) - 5} more parameters")
            raise RuntimeError(f"NaN detected in gradients at batch {batch_idx}. Stopping training.")

        # Optimizer step with gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping to prevent gradient explosion (critical for stability)
            if use_amp:
                # Unscale gradients before clipping when using AMP
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

            # NaN detection: Check model parameters after optimizer step
            nan_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params.append(name)
            if nan_params:
                print(f"[NaN Detection] Batch {batch_idx}: NaN detected in model parameters after optimizer step:")
                for name in nan_params[:5]:
                    print(f"  - {name}")
                if len(nan_params) > 5:
                    print(f"  ... and {len(nan_params) - 5} more parameters")
                raise RuntimeError(f"NaN detected in model parameters at batch {batch_idx}. Stopping training.")

        # Track recursion steps if available (HRM/TRM models)
        if "recursion_steps" in q_output:
            recursion_steps = q_output["recursion_steps"]
            total_recursion_steps += recursion_steps.sum().item()
            num_recursion_samples += recursion_steps.numel()

        # Accumulate losses (unscaled for logging)
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

    # Add average recursion steps if tracked
    if num_recursion_samples > 0:
        total_losses['avg_recursion_steps'] = total_recursion_steps / num_recursion_samples
    else:
        total_losses['avg_recursion_steps'] = 0.0

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
            elif config.get("model_type") == 'hrm':
                model = ChessNNet(config)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
            elif config.get("model_type") == 'trm':
                model = ChessTRMNet(config)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
            elif config.get("model_type") == 'trm_baseline':
                model = ChessTRMBaselineNet(config)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
            elif config.get("model_type") == 'transformer':
                model = TransformerChessNet(config)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
            else:
                raise ValueError(f"Unknown model type when loading checkpoint: {config.get('model_type')}")
            return model.to(device)
    if(config.get("model_type") == "simple"):
        return SimpleChessNet().to(device)
    if(config.get("model_type") == "hrm"):
        return ChessNNet(config).to(device)
    if(config.get("model_type") == "trm"):
        return ChessTRMNet(config).to(device)
    if(config.get("model_type") == "trm_baseline"):
        return ChessTRMBaselineNet(config).to(device)
    if(config.get("model_type") == "transformer"):
        return TransformerChessNet(config).to(device)

def save_model(model, optimizer, scheduler, config, args, epoch, losses):
    """Save model with training info"""
    if args.save_path:
        save_path = args.save_path
    else:
        # Create directory structure: model_checkpoints/{model_name}/epoch_{epoch}.pth
        model_name = config.get('name', 'unnamed_model')
        checkpoint_dir = f"model_checkpoints/{model_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = f"{checkpoint_dir}/epoch_{epoch}.pth"

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

    # Get chunk files
    chunk_files = glob.glob(os.path.join(args.data_path, "*.gz"))
    print(f'Found {len(chunk_files)} chunk files in {args.data_path}')
    
    if not chunk_files:
        print("No chunk files found! Make sure the data path is correct.")
        return

    # # Sample a specific number of random chunk files from the total available
    # #dataset used has nearly 20,000 chunk files, corresponding to ~2 million games
    # #to speed up training during experimentation, a smaller subset of 5000 chunk files is used
    # #coresponds to ~500,000 games
    num_chunks = 5
    # num_chunks = min(num_chunks, len(chunk_files))  # Don't exceed available files
    # random.seed(42) #set seed for reproducibility
    # chunk_files = random.sample(chunk_files, num_chunks)
    # random.seed()  # Reset seed for other random operations


    # chunk_files = chunk_files[:20]

    train_split = int(0.9 * len(chunk_files)) # 90% for training, 10% for validation

    print(f"[Main] Creating training dataset with {train_split} chunk files...")
    train_dataset = ChessDataset(chunk_files[:train_split], sample_rate=config.get('sample_rate', 0))  # Use higher sampling for speed
    print(f"[Main] Training dataset created successfully!")

    print(f"[Main] Creating validation dataset with {len(chunk_files) - train_split} chunk files...")
    valid_dataset = ChessDataset(chunk_files[train_split:], sample_rate=config.get('sample_rate', 0))
    print(f"[Main] Validation dataset created successfully!")

    # Optimize data loading with multiple workers and pinned memory
    # num_workers = config.get('num_workers', 16)  # Use 16 workers by default for parallel data loading
    num_workers = 16
    print(f"[Main] Creating DataLoaders with {num_workers} workers...")
    train_dataloader = DataLoader(train_dataset, batch_size=config.get('batch_size', 64),
                            shuffle=True, num_workers=num_workers, pin_memory=True,
                            persistent_workers=True if num_workers > 0 else False,
                            prefetch_factor=2 if num_workers > 0 else None)
    test_dataloader = DataLoader(valid_dataset, batch_size=config.get('batch_size', 64),
                            shuffle=False, num_workers=num_workers, pin_memory=True,
                            persistent_workers=True if num_workers > 0 else False,
                            prefetch_factor=2 if num_workers > 0 else None)
    print(f"[Main] DataLoaders created successfully!")

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
    # Add learning rate scheduler - cosine annealing (smooth decay, no restarts)
    #use T_max from config or default to 20 (realistic for early stopping)
    t_max = config.get('scheduler_T_max', 20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=config.get('lr', 0.0001)/100
    )
    print(f'Using CosineAnnealingLR: {config.get("lr", 0.0001):.2e} -> {config.get("lr", 0.0001)/100:.2e} over {t_max} epochs')

    # Initialize mixed precision scaler for AMP (if enabled and on CUDA)
    # Note: Don't use gradient scaler with bfloat16 (not supported)
    # bfloat16 provides mixed precision benefits without gradient scaling
    use_bfloat16 = (config.get('model_type') in ['hrm', 'trm'] and
                    config.get(f'{config.get("model_type")}_config', {}).get('forward_dtype') == 'bfloat16')
    use_amp = config.get('use_amp', False) and device.type == 'cuda' and not use_bfloat16
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

    if use_amp:
        print(f'Using Automatic Mixed Precision (AMP) training with float16')
    if use_bfloat16:
        print(f'Using bfloat16 mixed precision (no gradient scaling)')
    if gradient_accumulation_steps > 1:
        print(f'Using gradient accumulation with {gradient_accumulation_steps} steps')
        print(f'Effective batch size: {config.get("batch_size", 64) * gradient_accumulation_steps}')

    # Initialize TensorBoard writer
    log_dir = f"runs/chess_training_{config.get('name')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f'TensorBoard logs will be saved to: {log_dir}')

    # Early stopping parameters (hardcoded)
    early_stopping_patience = 4  # Stop if no improvement for 4 validation checks (set to None to disable)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    if early_stopping_patience:
        print(f'Early stopping enabled with patience: {early_stopping_patience} validation checks')

    # Training loop
    for epoch in range(config.get('epochs')):
        print(f'\nEpoch {epoch+1}/{config.get('epochs')}')

        losses = train_epoch(model, train_dataloader, criterion, optimizer, device,
                           scaler=scaler, gradient_accumulation_steps=gradient_accumulation_steps)
        
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

            # Early stopping check
            current_val_loss = eval_losses['total_loss']
            if early_stopping_patience:
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    epochs_without_improvement = 0
                    print(f'Validation loss improved to {best_val_loss:.6f}')
                else:
                    epochs_without_improvement += 1
                    print(f'No improvement for {epochs_without_improvement} validation checks (patience: {early_stopping_patience})')

                    if epochs_without_improvement >= early_stopping_patience:
                        print(f'\nEarly stopping triggered! No improvement for {early_stopping_patience} validation checks.')
                        print(f'Best validation loss: {best_val_loss:.6f}')
                        save_model(model, optimizer, scheduler, config, args, epoch, losses)
                        writer.close()
                        return

            # Save model checkpoint
            save_model(model, optimizer, scheduler, config, args, epoch, losses)

    # Close TensorBoard writer
    writer.close()
    print(f"Training completed. View logs with: tensorboard --logdir {log_dir}")
    # print(f"Parameter snapshots saved to: {param_snapshot_dir}")
    # print(f"To visualize parameter evolution, run: python visualize_parameters.py --snapshot-dir {param_snapshot_dir}")

if __name__ == '__main__':
    main()