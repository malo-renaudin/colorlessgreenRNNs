#!/bin/bash
#SBATCH --job-name=nounpp_rnn_relu # Job name
#SBATCH --partition=gpu 
#SBATCH --export=ALL 
#SBATCH --cpus-per-task=6        # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=32G                # Request 32GB of memory (adjust as needed)
#SBATCH --time=48:00:00          # Maximum job runtime (adjust as needed)
#SBATCH --output=log/%x-%j.log

module load miniconda3/24.3.0
module load python/3.9.18-lpwk
module load cuda/11.8.0-r465 
source ~/.bashrc
conda activate leaps3


echo "Running job on $(hostname)"
which python
echo "python-version $(python --version)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"
conda list 
 
python extract_predictions_several_check.py --checkpoints '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/rnn_relu' -i '/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp' --output '/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/results.csv'