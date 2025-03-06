#!/bin/bash
#SBATCH --job-name=test_val # Job name
#SBATCH --partition=gpu 
#SBATCH --export=ALL 
#SBATCH --cpus-per-task=6        # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=16G                # Request 16GB of memory (adjust as needed)
#SBATCH --time=03:00:00          # Maximum job runtime (adjust as needed)
#SBATCH --output=log/%x-%j.log



echo "Running job on $(hostname)"
echo "python: $(which python)"
echo "python-version $(python -V)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"

module load miniconda3/24.3.0

source activate leaps
 
python src/language_models/main.py --data english_data --name test_checkpointing --epochs 6 --cuda
