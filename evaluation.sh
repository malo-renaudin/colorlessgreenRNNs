#!/bin/bash
#SBATCH --job-name=evaluation # Job name
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
# Directory containing your model checkpoints
#!/bin/bash

# Directory containing the checkpoint files
CHECKPOINT_DIR="/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/full_check"

# Iterate over all checkpoint files in the directory
for CHECKPOINT_FILE in "$CHECKPOINT_DIR"/*; do
    if [[ -f "$CHECKPOINT_FILE" ]]; then  # Check if it is a file
        # Get the base filename (without the path)
        FILENAME=$(basename "$CHECKPOINT_FILE")

        # Use the filename as the suffix
        SUFFIX="$FILENAME"

        # Run the Python script on each checkpoint file
        python /scratch2/mrenaudin/colorlessgreenRNNs/src/language_models/evaluate_target_word.py --data "/scratch2/mrenaudin/colorlessgreenRNNs/english_data/" --checkpoint "$CHECKPOINT_FILE" --path "/scratch2/mrenaudin/colorlessgreenRNNs/data/agreement/English/generated" --suffix "$SUFFIX" --cuda
    fi
done
