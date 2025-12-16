#!/usr/bin/env python3
"""
Batch training script for multiple chess models.
Automatically trains all model configurations sequentially.
"""

import os
import sys
import glob
import argparse
import yaml
import subprocess
from datetime import datetime
from pathlib import Path


def find_config_files(config_dir):
    """Find all YAML config files in the specified directory and subdirectories."""
    config_files = []

    # Search for .yaml and .yml files
    for ext in ['*.yaml', '*.yml']:
        config_files.extend(glob.glob(os.path.join(config_dir, '**', ext), recursive=True))

    # Sort for consistent ordering
    config_files.sort()
    return config_files


def load_config(config_path):
    """Load and return the config from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_model(config_path, data_path, device='cuda', model_path=None):
    """
    Train a single model with the given configuration.

    Args:
        config_path: Path to the YAML config file
        data_path: Path to training data chunks
        device: Device to use for training (cuda/cpu)
        model_path: Optional path to existing model checkpoint to continue training

    Returns:
        bool: True if training succeeded, False otherwise
    """
    print("\n" + "="*80)
    print(f"Starting training with config: {config_path}")
    print("="*80 + "\n")

    # Build the command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        'pytorch_train.py',
        '--config', config_path,
        '--data-path', data_path,
        '--device', device
    ]

    # Add model path if provided
    if model_path:
        cmd.extend(['--model-path', model_path])

    # Run the training
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Successfully completed training for: {config_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed for: {config_path}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Batch train multiple chess models sequentially',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all configs in config/ directory
  python batch_train.py --config-dir config/

  # Train specific configs
  python batch_train.py --configs config/hrm/hrm_halt4.yaml config/trm/trm_halt1.yaml

  # Train all configs with custom data path
  python batch_train.py --config-dir config/ --data-path data/my-training-data

  # Continue training with existing checkpoints
  python batch_train.py --config-dir config/ --resume
        """
    )

    # Config selection
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--config-dir',
        type=str,
        help='Directory containing config files (will train all configs found recursively)'
    )
    config_group.add_argument(
        '--configs',
        type=str,
        nargs='+',
        help='Specific config file paths to train'
    )

    # Training parameters
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/training-run3--20210605-0521',
        help='Path to training data chunks (default: data/training-run3--20210605-0521)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for training (default: cuda)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from existing checkpoints if available'
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop batch training if any model fails (default: continue to next model)'
    )

    args = parser.parse_args()

    # Get list of config files to train
    if args.config_dir:
        if not os.path.isdir(args.config_dir):
            print(f"Error: Config directory not found: {args.config_dir}")
            return 1
        config_files = find_config_files(args.config_dir)
        if not config_files:
            print(f"Error: No config files found in: {args.config_dir}")
            return 1
    else:
        config_files = args.configs
        # Verify all config files exist
        for config_file in config_files:
            if not os.path.isfile(config_file):
                print(f"Error: Config file not found: {config_file}")
                return 1

    # Skip first 2 configs (already trained)
    if len(config_files) > 2:
        print(f"\nSkipping first 2 configs (already trained):")
        for i, config_file in enumerate(config_files[:2], 1):
            try:
                config = load_config(config_file)
                model_name = config.get('name', 'unnamed')
                print(f"  {i}. {config_file} → {model_name}")
            except:
                print(f"  {i}. {config_file}")
        config_files = config_files[2:]
        print(f"Remaining configs to train: {len(config_files)}\n")
    else:
        print(f"Warning: Only {len(config_files)} config(s) found, not skipping any.\n")

    # Print summary
    print("\n" + "="*80)
    print("BATCH TRAINING SUMMARY")
    print("="*80)
    print(f"Total configs to train: {len(config_files)}")
    print(f"Data path: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"Resume from checkpoints: {args.resume}")
    print(f"Stop on error: {args.stop_on_error}")
    print("\nConfigs to train:")
    for i, config_file in enumerate(config_files, 1):
        try:
            config = load_config(config_file)
            model_name = config.get('name', 'unnamed')
            model_type = config.get('model_type', 'unknown')
            epochs = config.get('epochs', '?')
            print(f"  {i}. {config_file}")
            print(f"     → Name: {model_name}, Type: {model_type}, Epochs: {epochs}")
        except Exception as e:
            print(f"  {i}. {config_file} (failed to load: {e})")
    print("="*80 + "\n")

    # Ask for confirmation
    try:
        response = input("Proceed with batch training? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Batch training cancelled")
            return 0
    except KeyboardInterrupt:
        print("\nBatch training cancelled")
        return 0

    # Track results
    start_time = datetime.now()
    results = []

    # Train each model
    for i, config_file in enumerate(config_files, 1):
        print(f"\n\nProgress: {i}/{len(config_files)}")

        # Determine model path if resuming
        model_path = None
        if args.resume:
            try:
                config = load_config(config_file)
                model_name = config.get('name', 'unnamed_model')
                checkpoint_dir = f"model_checkpoints/{model_name}"

                # Find the latest checkpoint
                if os.path.exists(checkpoint_dir):
                    checkpoints = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
                    if checkpoints:
                        # Sort by epoch number
                        checkpoints.sort(key=lambda x: int(x.split('epoch_')[-1].split('.pth')[0]))
                        model_path = checkpoints[-1]
                        print(f"Resuming from checkpoint: {model_path}")
            except Exception as e:
                print(f"Warning: Could not find checkpoint to resume from: {e}")

        # Train the model
        try:
            success = train_model(config_file, args.data_path, args.device, model_path)
            results.append((config_file, success))

            if not success and args.stop_on_error:
                print("\nStopping batch training due to error (--stop-on-error enabled)")
                break

        except KeyboardInterrupt:
            print("\n\nBatch training interrupted by user")
            results.append((config_file, False))
            break

    # Print final summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n\n" + "="*80)
    print("BATCH TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {duration}")
    print(f"Configs trained: {len(results)}/{len(config_files)}")
    print(f"Successful: {sum(1 for _, success in results if success)}")
    print(f"Failed: {sum(1 for _, success in results if not success)}")

    print("\nResults:")
    for config_file, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {config_file}")

    if len(results) < len(config_files):
        print(f"\nNot completed: {len(config_files) - len(results)} configs")

    print("="*80 + "\n")

    # Return exit code based on results
    return 0 if all(success for _, success in results) else 1


if __name__ == '__main__':
    sys.exit(main())