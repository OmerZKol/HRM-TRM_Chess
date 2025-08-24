"""
Chess move prediction evaluation script.
"""

import os
import json
import torch
from typing import List, Dict, Tuple

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.losses import ACTLossHead
from dataset.build_chess_dataset import ChessMoveEncoder, create_position_encoding


class ChessPredictor:
    """Chess move prediction using trained HRM model."""
    
    def __init__(self, checkpoint_path: str, dataset_path: str):
        torch.set_default_device("cuda")
        self.device = torch.device("cuda")  # Use GPU for inference
        self.encoder = ChessMoveEncoder()
        
        # Load dataset info
        with open(os.path.join(dataset_path, "dataset.json"), 'r') as f:
            self.dataset_info = json.load(f)
        
        # Load model
        self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint."""
        print(f"Loading model from {checkpoint_path}...")
        
        # Model config (should match training config)
        model_config = {
            "batch_size": 1,  # Inference batch size
            "seq_len": self.dataset_info["seq_len"],
            "vocab_size": self.dataset_info["vocab_size"],
            "num_puzzle_identifiers": 1000,
            
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
            
            # Chess move prediction config
            "use_move_prediction": True,
            "num_actions": self.dataset_info["num_actions"],
            "move_prediction_from_token": 0,
            
            # Puzzle embeddings
            "puzzle_emb_ndim": 512,
            
            # Training dtype
            "forward_dtype": "bfloat16"
        }
        
        # Create model
        model = HierarchicalReasoningModel_ACTV1(model_config)
        loss_model = ACTLossHead(
            model=model,
            loss_type="softmax_cross_entropy",
            move_loss_weight=2.0
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        loss_model.load_state_dict(checkpoint["model_state_dict"])
        
        self.model = loss_model.to(self.device)
        self.model.eval()
        
        # Ensure all buffers and parameters are on the correct device
        def move_to_device(module):
            for name, buffer in module.named_buffers():
                buffer.data = buffer.data.to(self.device)
            for name, param in module.named_parameters():
                param.data = param.data.to(self.device)
        
        move_to_device(self.model)
        
        print(f"Model loaded successfully!")
    
    @torch.no_grad()
    def predict_move(self, move_history: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict next chess move given move history.
        
        Args:
            move_history: List of moves in UCI format (e.g., ['e2e4', 'd7d5', ...])
            top_k: Number of top moves to return
            
        Returns:
            List of (move, probability) tuples
        """
        # Create position encoding
        position_encoding = create_position_encoding(move_history, self.dataset_info["seq_len"])
        
        # Create batch
        batch = {
            "inputs": torch.tensor([position_encoding], dtype=torch.long).to(self.device),
            "labels": torch.full((1, len(position_encoding)), -100, dtype=torch.long).to(self.device),
            "move_targets": torch.tensor([-100], dtype=torch.long).to(self.device),
            "puzzle_identifiers": torch.tensor([0], dtype=torch.long).to(self.device)
        }
        
        # Initialize carry
        carry = self.model.initial_carry(batch)
        
        # Forward pass
        carry, loss, metrics, outputs, all_halted = self.model.forward(
            carry=carry,
            batch=batch,
            return_keys=["move_logits"]
        )
        
        # Get move probabilities
        if "move_logits" in outputs:
            move_logits = outputs["move_logits"][0]  # Remove batch dimension
            move_probs = torch.softmax(move_logits, dim=-1)
            
            # Get top-k moves
            top_probs, top_indices = torch.topk(move_probs, top_k)
            
            # Convert to move strings
            top_moves = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                move_str = self.encoder.decode_move(idx.item())
                top_moves.append((move_str, prob.item()))
            
            return top_moves
        else:
            print("Warning: No move logits in output!")
            return []
    
    def analyze_position(self, move_history: List[str]) -> Dict:
        """Analyze a chess position and return detailed predictions."""
        top_moves = self.predict_move(move_history, top_k=10)
        
        analysis = {
            "position": {
                "move_history": move_history,
                "moves_played": len(move_history),
                "to_move": "white" if len(move_history) % 2 == 0 else "black"
            },
            "predictions": top_moves,
            "top_move": top_moves[0] if top_moves else None
        }
        
        return analysis


def demo_prediction():
    """Demo chess move prediction."""
    # Example: Load model and make predictions
    checkpoint_path = "final_checkpoint.pt"  # Update with actual path
    dataset_path = "data/chess-move-prediction"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using train_chess.py")
        return
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Please build the dataset first using dataset/build_chess_dataset.py")
        return
    
    # Initialize predictor
    predictor = ChessPredictor(checkpoint_path, dataset_path)
    
    # Example positions
    test_positions = [
        # Opening
        [],
        ["e2e4"],
        ["e2e4", "e7e5"],
        ["e2e4", "e7e5", "g1f3"],
        
        # Mid-game
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"],
    ]
    
    print("Chess Move Prediction Demo")
    print("=" * 40)
    
    for i, moves in enumerate(test_positions):
        print(f"\nPosition {i+1}:")
        print(f"Moves: {' '.join(moves) if moves else '(starting position)'}")
        
        analysis = predictor.analyze_position(moves)
        
        print(f"To move: {analysis['position']['to_move']}")
        print("Top predicted moves:")
        for j, (move, prob) in enumerate(analysis['predictions'][:5], 1):
            print(f"  {j}. {move} ({prob:.1%})")


if __name__ == "__main__":
    demo_prediction()