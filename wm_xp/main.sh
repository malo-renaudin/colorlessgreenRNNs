#!/bin/bash

# Simple script to submit experiments without SLURM for the setup part
# Usage: ./submit_experiment.sh <experiment_name> "<hidden_dims>" "<num_heads>" "<temperatures>" "<gumbel_options>"

EXPERIMENT_NAME="$1"
HIDDEN_DIMS="$2"
NUM_HEADS="$3"
TEMPERATURES="$4"
GUMBEL_OPTIONS="$5"

# Check if all required arguments are provided
if [ -z "$EXPERIMENT_NAME" ] || [ -z "$HIDDEN_DIMS" ] || [ -z "$NUM_HEADS" ] || [ -z "$TEMPERATURES" ] || [ -z "$GUMBEL_OPTIONS" ]; then
    echo "Usage: $0 <experiment_name> \"<hidden_dims>\" \"<num_heads>\" \"<temperatures>\" \"<gumbel_options>\""
    echo "Example: $0 my_exp \"128 256 512\" \"4 8 16\" \"0.5 1.0 2.0\" \"true false\""
    exit 1
fi

# Load environment (adjust paths as needed)
source cgr/bin/activate

# Create logs directory
mkdir -p logs

echo "Setting up experiment: $EXPERIMENT_NAME"
echo "Hidden dimensions: $HIDDEN_DIMS"
echo "Number of heads: $NUM_HEADS"
echo "Temperatures: $TEMPERATURES"
echo "Gumbel options: $GUMBEL_OPTIONS"

echo "Running experiment setup..."

# Run the experiment runner with proper argument format for xp_parser
python wm_xp/runner.py \
    --name "$EXPERIMENT_NAME" \
    --data english_data \
    --hidden_dims $HIDDEN_DIMS \
    --num_heads $NUM_HEADS \
    --temperatures $TEMPERATURES \
    --gumbel_softmax $GUMBEL_OPTIONS \
    --grid_search_script wm_xp/gs_eval.sh \
    --eval_script wm_xp/tests/tests.py \
    --nounpp NounPP/Stimuli/nounpp.txt \
    --cuda

if [ $? -eq 0 ]; then
    echo "✓ Experiment setup completed successfully!"
    echo "✓ Grid search jobs submitted to SLURM"
    echo "Monitor with: squeue -u \$USER"
    echo "Results will be in: wm_xp/experiments/$EXPERIMENT_NAME"
else
    echo "❌ Experiment setup failed!"
    exit 1
fi