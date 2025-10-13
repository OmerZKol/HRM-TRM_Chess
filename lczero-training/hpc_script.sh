#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=82G
#SBATCH --output=/users/acb22av/HRM-Chess/lczero-training/hpc-output/%j-%a.out
#SBATCH --error=/users/acb22av/HRM-Chess/lczero-training/hpc-output/%j-%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=avakhutinskiy1@sheffield.ac.uk
#SBATCH --job-name=hrm-chess
#SBATCH --array=1-4

module load Anaconda3/2024.02-1
module load cuDNN
module load GCC

cd /users/acb22av/HRM-Chess/lczero-training
PYTHON_BIN="/users/acb22av/.conda/envs/hrm-chess/bin/python3.12"

mkdir -p hpc-output

export PYTHONUNBUFFERED=1

case $SLURM_ARRAY_TASK_ID in
    1)
        CONFIG="config/simple_chess_nn.yaml"
        MODEL_NAME="simple_cnn"
        ;;
    2)
        CONFIG="config/transformer_chess_nn.yaml"
        MODEL_NAME="transformer"
        ;;
    3)
        CONFIG="config/hrm_halt1.yaml"
        MODEL_NAME="hrm_halt1"
        ;;
    4)
        CONFIG="config/hrm_halt10.yaml"
        MODEL_NAME="hrm_halt10"
        ;;
esac

echo "=========================================="
echo "Starting training for: $MODEL_NAME"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Run training with full path to Python interpreter
$PYTHON_BIN pytorch_train.py \
    --config $CONFIG \
    --data-path data/training-run1--20250209-1017