#!/bin/bash
#SBATCH --job-name=evaluation2 # Job name
#SBATCH --partition=gpu 
#SBATCH --export=ALL 
#SBATCH --cpus-per-task=6        # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=16G                # Request 32GB of memory (adjust as needed)
#SBATCH --time=30:00:00          # Maximum job runtime (adjust as needed)
#SBATCH --output=log/%x-%j.log

module load miniconda3/24.3.0
module load python/3.9.18-lpwk
module load cuda/11.8.0-r465 
source ~/.bashrc
conda activate leaps3

python /scratch2/mrenaudin/colorlessgreenRNNs/src/results2.py