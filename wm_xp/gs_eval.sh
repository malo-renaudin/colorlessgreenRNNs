#!/bin/bash
#SBATCH --job-name=grid_search_eval
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err
#SBATCH --array=1-54
#SBATCH --time=48:00:00  
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --account=ywa@h100
#SBATCH --hint=nomultithread
#SBATCH --partition=gpu_p6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malorenaudin1@gmail.com
#SBATCH —cpus-per-task=24


# Parse command line arguments
CONFIG_FILE="$1"
OUTPUT_BASE_DIR="$2"
EVAL_SCRIPT_PATH="$3"  # Path to your evaluation script

# Load necessary modules
# module load miniconda3/24.3.0
# module load python/3.9.18-lpwk
# module load cuda/11.8.0-r465
# source ~/.bashrc

# conda activate leaps3
source cgr/bin/activate
# Create output directories
mkdir -p "$OUTPUT_BASE_DIR"

# Get the parameter combination for this array task
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $CONFIG_FILE)

# Parse parameters (comma-separated)
IFS=',' read -r hidden_dim num_heads temperature gumbel_softmax <<< "$PARAMS"

# Create config-based output directory
CONFIG_DIR="h${hidden_dim}_heads${num_heads}_t${temperature}_${gumbel_softmax}"
JOB_OUTPUT_DIR="$OUTPUT_BASE_DIR/$CONFIG_DIR"
mkdir -p "$JOB_OUTPUT_DIR"

# Create experiment name and gumbel flag based on parameters
if [ "$gumbel_softmax" = "true" ]; then
    exp_name="h${hidden_dim}_heads${num_heads}_t${temperature}_gumbel"
    gumbel_flag="--gumbel_softmax"
    gumbel_eval_flag="--gumbel"
else
    exp_name="h${hidden_dim}_heads${num_heads}_t${temperature}_nogumbel"
    gumbel_flag=""
    gumbel_eval_flag=""
fi

echo "=== STARTING TRAINING ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Parameters: hidden_dim=${hidden_dim}, num_heads=${num_heads}, temperature=${temperature}, gumbel_softmax=${gumbel_softmax}"
echo "Experiment name: $exp_name"
echo "Output directory: $JOB_OUTPUT_DIR"

# Run training
python src/language_models/main.py \
    --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data \
    --name "$exp_name" \
    --classmodel 'CBR_RNN' \
    --batch_size 512 \
    --emsize "$hidden_dim" \
    --nhid "$hidden_dim" \
    --nheads "$num_heads" \
    --min_temp "$temperature" \
    $gumbel_flag \
    --optimizer 'Adam' \
    --epochs 40 \
    --cuda \
    --checkpoint_dir "$JOB_OUTPUT_DIR"

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Training failed for config: $exp_name"
    exit 1
fi

echo "=== TRAINING COMPLETED ==="
echo "=== STARTING EVALUATION ==="

# Find the checkpoint file
CHECKPOINT_PATH="${JOB_OUTPUT_DIR}/${exp_name}/epoch_40.pt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
    exit 1
fi

# Create evaluation output directory
EVAL_OUTPUT_DIR="$JOB_OUTPUT_DIR/evaluation"
mkdir -p "$EVAL_OUTPUT_DIR"

# Run all evaluations
echo "Running comprehensive evaluation (NounPP + BLiMP + Repeat Surprisal)..."
python "$EVAL_SCRIPT_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_path /scratch2/mrenaudin/colorlessgreenRNNs/english_data \
    --hidden_dim "$hidden_dim" \
    --nheads "$num_heads" \
    $gumbel_eval_flag \
    --output_dir "$EVAL_OUTPUT_DIR" \
    --nounpp /scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt\
    --cuda

if [ $? -eq 0 ]; then
    echo "All evaluations completed successfully"
else
    echo "Evaluation failed - check error logs"
fi

echo "=== ALL EVALUATIONS COMPLETED ==="
echo "Results saved in: $EVAL_OUTPUT_DIR"
echo "Checkpoint saved at: $CHECKPOINT_PATH"