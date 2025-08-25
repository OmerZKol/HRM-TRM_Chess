#!/usr/bin/env python3
"""
Complete pipeline to train HRM for chess move prediction.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"✓ {description} completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train HRM for chess move prediction")
    parser.add_argument("--csv-path", default="data/chess_games_more_filtered.csv",
                       help="Path to chess games CSV file")
    parser.add_argument("--max-games", type=int, default=10000,
                       help="Maximum number of games to process")
    parser.add_argument("--min-elo", type=int, default=2200,
                       help="Minimum ELO rating for games")
    parser.add_argument("--skip-dataset", action="store_true",
                       help="Skip dataset building (if already exists)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    
    args = parser.parse_args()
    
    print("HRM Chess Move Prediction Training Pipeline")
    # print(f"Max games: {args.max_games}")
    # print(f"Min ELO: {args.min_elo}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # # Check if chess games CSV exists
    # if not os.path.exists(args.csv_path):
    #     print(f"ERROR: Chess games CSV not found at {args.csv_path}")
    #     print("Please ensure the chess games file exists.")
    #     sys.exit(1)
    
    # # Step 1: Build dataset
    # if not args.skip_dataset:
    #     dataset_cmd = f"""python dataset/build_chess_dataset.py \
    #         --csv-path {args.csv_path} \
    #         --output-dir data/chess-move-prediction \
    #         --max-games {args.max_games} \
    #         --min-elo {args.min_elo} \
    #         --max-moves-per-game 40"""
        
    #     run_command(dataset_cmd, "Building chess dataset")
    # else:
    #     print("Skipping dataset building...")
    #     if not os.path.exists("data/chess-move-prediction"):
    #         print("ERROR: Dataset directory not found and --skip-dataset was specified")
    #         sys.exit(1)
    
    # Step 2: Train model
    # Update config with command line arguments
    config_overrides = [
        f"epochs={args.epochs}",
        f"global_batch_size={args.batch_size}",
        "data_path=data/chess-move-prediction",
    ]
    
    train_cmd = f"python train_chess.py {' '.join(config_overrides)}"
    run_command(train_cmd, "Training HRM chess model")
    
    # Step 3: Test the model
    if os.path.exists("final_checkpoint.pt"):
        print("\n" + "="*60)
        print("Testing trained model...")
        print("="*60)
        
        test_cmd = "python evaluate_chess.py"
        try:
            subprocess.run(test_cmd, shell=True, check=True)
            print("✓ Model testing completed!")
        except subprocess.CalledProcessError:
            print("⚠ Model testing failed, but training was successful")
    
    print("\n" + "="*60)
    print("🎉 Chess training pipeline completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check wandb logs for training metrics")
    print("2. Use evaluate_chess.py to test specific positions")
    print("3. Fine-tune hyperparameters if needed")
    print("\nModel files:")
    print("- final_checkpoint.pt: Trained model")
    print("- data/chess-move-prediction/: Dataset files")


if __name__ == "__main__":
    main()