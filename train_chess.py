"""
Training script for HRM chess move prediction.
"""

import os
import math
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.losses import ACTLossHead
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from chess_puzzle_dataset import ChessPuzzleDataset, ChessPuzzleDatasetConfig

def get_parameter_count(model: torch.nn.Module) -> int:
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass 
class ChessTrainingConfig:
    # Data
    data_path: str = "data/chess-move-prediction"
    
    # Training
    epochs: int = 50
    global_batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Model specific
    puzzle_emb_lr: Optional[float] = None
    puzzle_emb_weight_decay: Optional[float] = None
    
    # Evaluation
    print_interval: int = 100
    eval_interval: int = 3000
    save_interval: int = 10000
    
    # Logging
    project_name: str = "hrm-chess"
    run_name: Optional[str] = None
    
    # System
    device: str = "cuda"
    compile: bool = False
    mixed_precision: bool = True
    
    # Distributed training
    rank: int = 0
    world_size: int = 1


class ChessTrainer:
    def __init__(self, config: ChessTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.step = 0
        self.epoch = 0
        self.use_wandb = False
        self.checkpoint_path = "checkpoints/"
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        # Load dataset metadata
        with open(os.path.join(config.data_path, "dataset.json"), 'r') as f:
            self.dataset_info = json.load(f)
        
        print(f"Dataset info: {self.dataset_info}")
        
        # Initialize model
        self._init_model()
        
        # Initialize datasets
        self._init_datasets()
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Initialize logging
        if config.rank == 0:
            self._init_logging()
        
        # Mixed precision
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _init_model(self):
        """Initialize HRM model with chess move prediction."""
        model_config = {
            # Required HRM config
            "batch_size": self.config.global_batch_size,
            "seq_len": self.dataset_info["seq_len"],
            "vocab_size": self.dataset_info["vocab_size"],
            "num_puzzle_identifiers": 1,  # Group games by puzzle ID
            
            # Architecture
            "H_cycles": 3,
            "L_cycles": 2,
            "H_layers": 4,
            "L_layers": 4,
            "hidden_size": 512,
            "num_heads": 8,
            "expansion": 4.0,
            "pos_encodings": "rope",
            
            # ACT config
            "halt_max_steps": 8,
            "halt_exploration_prob": 0.1,
            
            # Chess move prediction config (NEW)
            "use_move_prediction": True,
            "num_actions": self.dataset_info["num_actions"],
            "move_prediction_from_token": 0,  # Use first token

            # Chess value prediction
            "use_value_prediction": True,

            # Use linear projection to tokenize each square
            "use_chess_tokenization": True,
            "square_feature_dim": 112,  # 112 features per square

            # Puzzle embeddings
            "puzzle_emb_ndim": 512,
            
            # Training dtype
            "forward_dtype": "bfloat16"
        }
        self.model_config = model_config
        
        print(f"Model config: {model_config}")
        
        # Create model
        self.model = HierarchicalReasoningModel_ACTV1(model_config)
        
        # Wrap with loss head
        self.loss_model = ACTLossHead(
            model=self.model,
            loss_type="softmax_cross_entropy",
            move_loss_weight=2.0  # Higher weight for move prediction
        )
        
        self.loss_model = self.loss_model.to(self.device)
        
        # Compile if requested
        if self.config.compile:
            print("Compiling model...")
            self.loss_model = torch.compile(self.loss_model)
        
        # Print model info
        total_params = get_parameter_count(self.loss_model)
        print(f"Model parameters: {total_params:,}")
    
    def _init_datasets(self):
        """Initialize chess datasets."""
        dataset_config = ChessPuzzleDatasetConfig(
            seed=42,
            dataset_path=self.config.data_path,
            global_batch_size=self.config.global_batch_size,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=self.config.rank,
            num_replicas=self.config.world_size
        )
        
        # Training dataset
        self.train_dataset = ChessPuzzleDataset(dataset_config, split="train")
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=None,  # Already batched
            num_workers=0,    # Single-threaded
            pin_memory=True
        )
        
        # Validation dataset
        eval_config = ChessPuzzleDatasetConfig(
            seed=42,
            dataset_path=self.config.data_path,
            global_batch_size=self.config.global_batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=self.config.rank,
            num_replicas=self.config.world_size
        )
        
        self.val_dataset = ChessPuzzleDataset(eval_config, split="test")
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=True
        )
    
    def _init_optimizer(self):
        """Initialize optimizer with different learning rates for puzzle embeddings."""
        # Separate puzzle embedding parameters
        puzzle_emb_params = []
        other_params = []
        puzzle_param_names = []
        if hasattr(self.model.inner, 'puzzle_emb') and self.model.inner.puzzle_emb is not None:
            for name, param in self.model.inner.puzzle_emb.named_parameters():
                puzzle_emb_params.append(param)
                # The loss_model prepends "model.inner." to the name
                puzzle_param_names.append(f"model.inner.puzzle_emb.{name}")

        for name, param in self.loss_model.named_parameters():
            if param.requires_grad:
                if name not in puzzle_param_names:
                    other_params.append(param)
        
        # Create parameter groups
        main_param_groups = [
            {
                "params": other_params,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay
            }
        ]

        self.optimizer = torch.optim.AdamW(main_param_groups)

        self.puzzle_optimizer = None
        if puzzle_emb_params:
            # This dummy optimizer is only for the GradScaler to find and unscale the puzzle_emb gradients
            self.dummy_puzzle_optimizer_for_scaler = torch.optim.SGD(puzzle_emb_params, lr=0)
            self.puzzle_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
                params=puzzle_emb_params,
                lr=self.config.puzzle_emb_lr or self.config.lr,
                weight_decay=self.config.puzzle_emb_weight_decay or self.config.weight_decay,
                world_size=self.config.world_size
            )

        # Learning rate scheduler for the main optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs * 1000,  # Approximate steps per epoch
            eta_min=self.config.lr * 0.1
        )
        
    def _init_logging(self):
        """Initialize Weights & Biases logging."""
        try:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=self.config.__dict__
            )
            self.use_wandb = True
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")
            self.use_wandb = False
    
    def _get_lr_multiplier(self) -> float:
        """Get learning rate multiplier for warmup."""
        if self.step < self.config.warmup_steps:
            return self.step / self.config.warmup_steps
        return 1.0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step."""
        self.loss_model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        # Initialize carry
        carry = self.loss_model.initial_carry(batch)
        
        # Forward pass with mixed precision
        # Disable mixed precision when using BFloat16 due to CUDA compatibility issues
        use_mixed_precision = self.config.mixed_precision and self.model_config.get("forward_dtype", "float32") != "bfloat16"
        with torch.amp.autocast('cuda', enabled=use_mixed_precision):
            carry, loss, metrics, outputs, all_halted = self.loss_model.forward(
                carry=carry,
                batch=batch,
                return_keys=["move_logits"]
            )
        
        # Backward pass
        if self.scaler and use_mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.puzzle_optimizer:
                self.scaler.unscale_(self.dummy_puzzle_optimizer_for_scaler) # Unscale puzzle grads
            grad_norm = clip_grad_norm_(self.loss_model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            if self.puzzle_optimizer:
                self.puzzle_optimizer.step() # Now operates on unscaled grads
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = clip_grad_norm_(self.loss_model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            if self.puzzle_optimizer:
                self.puzzle_optimizer.step()
        
        self.optimizer.zero_grad()
        if self.puzzle_optimizer:
            self.puzzle_optimizer.zero_grad()
        
        # Apply learning rate warmup
        lr_mult = self._get_lr_multiplier()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.lr * lr_mult
        
        self.scheduler.step()
        self.step += 1
        
        # Return metrics
        metrics_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
        metrics_dict.update({
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "lr": self.optimizer.param_groups[0]['lr'],
            "step": self.step,
        })
        
        return metrics_dict
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.loss_model.eval()
        
        total_loss = 0.0
        total_samples = 0
        total_metrics = {}
        for set_name, batch, batch_size in self.val_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            # Initialize carry
            carry = self.loss_model.initial_carry(batch)
            
            # Forward pass
            carry, loss, metrics, outputs, all_halted = self.loss_model.forward(
                carry=carry,
                batch=batch,
                return_keys=["move_logits"]
            )
            
            # Accumulate metrics
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for k, v in metrics.items():
                if torch.is_tensor(v):
                    v = v.item()
                total_metrics[k] = total_metrics.get(k, 0.0) + v

            if total_samples >= 500:
                break

        # Average metrics
        eval_metrics = {
            "eval_loss": total_loss / total_samples if total_samples > 0 else 0.0,
        }
        
        for k, v in total_metrics.items():
            eval_metrics[f"eval_{k}"] = v / len(list(self.val_loader)) if total_metrics else 0.0
        
        return eval_metrics
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.loss_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "config": self.config.__dict__
        }
        if self.puzzle_optimizer:
            checkpoint["puzzle_optimizer_state_dict"] = self.puzzle_optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.epochs} epochs...")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            # Training loop
            for set_name, batch, batch_size in self.train_loader:
                metrics = self.train_step(batch)
                # Log metrics
                if self.config.rank == 0:
                    if self.step % self.config.print_interval == 0:
                        print(f"Step {self.step}: Loss={metrics['loss']:.4f}, "
                              f"Move Acc={metrics.get('move_accuracy', 0)}/{metrics.get('move_count', 1)} "
                              f"({100*metrics.get('move_accuracy', 0)/max(metrics.get('move_count', 1), 1):.1f}%)")
                    
                    if self.use_wandb and self.step % 10 == 0:
                        wandb.log(metrics, step=self.step)
                
            # Evaluation at the end of the epoch
            # if self.step % self.config.eval_interval == 0:
            eval_metrics = self.evaluate()
            if self.config.rank == 0:
                print(f"Evaluation at step {self.step}: {eval_metrics}")
                if self.use_wandb:
                    wandb.log(eval_metrics, step=self.step)
            
            # Save checkpoint at the end of the epoch
            # if self.step % self.config.save_interval == 0 and self.config.rank == 0:
            checkpoint_path = f"{self.checkpoint_path}checkpoint_epoch_{epoch}.pt"
            self.save_checkpoint(checkpoint_path)
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        # Final checkpoint
        if self.config.rank == 0:
            final_path = f"{self.checkpoint_path}final_checkpoint.pt"
            self.save_checkpoint(final_path)
        
        print("Training completed!")


@hydra.main(version_base=None, config_path="config", config_name="cfg_chess")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    torch.set_default_device('cuda')
    # Convert to training config
    config = ChessTrainingConfig(**cfg)

    # Initialize trainer
    trainer = ChessTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()