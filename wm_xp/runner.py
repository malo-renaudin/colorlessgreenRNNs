#!/usr/bin/env python3
"""
Integrated experiment runner script for training and evaluation.
"""
import os
import subprocess
import sys
from pathlib import Path
import argparse
from xp_argparser import xp_parser
from configs.generate_config_file import generate_config
import torch
import json
import time
from datetime import datetime

# Use your xp_parser directly
args = xp_parser.parse_args()
device = torch.device("cuda" if args.cuda else "cpu")

print("="*60)
print("INTEGRATED TRAINING + EVALUATION EXPERIMENT")
print("="*60)
print(f"Experiment name: {args.name}")
print(f"Data path: {args.data}")
print(f"Device: {device}")
print(f"Grid search script: {args.grid_search_script}")
print(f"Evaluation script: {args.eval_script}")
print("="*60)

# Create experiment folder structure
exp_name = args.name
exp_dir = Path(f"wm_xp/experiments/{exp_name}")
config_dir = exp_dir / "config"
results_dir = exp_dir / "results"  # Will contain both checkpoints and evaluations

# Create all directories
exp_dir.mkdir(parents=True, exist_ok=True)
config_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# Create config file in the config subfolder
config_file_path = config_dir / args.name
print("Creating hyperparameter configuration...")
generate_config(args.hidden_dims, args.num_heads, args.temperatures, args.gumbel_softmax, str(config_file_path))

# Count total configurations
with open(config_file_path, 'r') as f:
    total_configs = len(f.readlines())

print(f"Generated {total_configs} hyperparameter configurations")

# Create experiment metadata
metadata = {
    "experiment_name": exp_name,
    "total_configurations": total_configs,
    "hidden_dims": args.hidden_dims,
    "num_heads": args.num_heads,
    "temperatures": args.temperatures,
    "gumbel_softmax": args.gumbel_softmax,
    "data_path": args.data,
    "device": str(device),
    "grid_search_script": args.grid_search_script,
    "eval_script": args.eval_script,
    "nounpp_path": args.nounpp,
    "start_time": str(datetime.now()),
    "status": "running"
}

with open(exp_dir / "experiment_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print("Starting integrated training and evaluation...")
print(f"This will train {total_configs} models and evaluate each on:")
print("  - NounPP syntactic evaluation")
print("  - BLiMP linguistic acceptability")
print("  - Repeat surprisal working memory tests")
print("="*60)

start_time = time.time()

try:
    # Run integrated grid search (training + evaluation)
    # Pass the evaluation script as the third argument
    subprocess.run([
        "sbatch", 
        args.grid_search_script, 
        str(config_file_path), 
        str(results_dir),
        args.eval_script
    ], check=True)
    
    print("✓ Grid search job submitted successfully!")
    print(f"Monitor progress with: squeue -u $USER")
    print(f"Check logs in: {exp_dir}/logs/")
    print(f"Results will be saved in: {results_dir}/")
    
    # Update metadata
    metadata["status"] = "submitted"
    metadata["submission_time"] = str(datetime.now())
    
    with open(exp_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("="*60)
    print("EXPERIMENT SETUP COMPLETE")
    print("="*60)
    print(f"Experiment directory: {exp_dir}")
    print(f"Configuration file: {config_file_path}")
    print(f"Results directory: {results_dir}")
    print(f"Expected structure per config:")
    print(f"  results_dir/h<dim>_heads<n>_t<temp>_<gumbel>/")
    print(f"    ├── model_checkpoint.pt")
    print(f"    └── evaluation/")
    print(f"        ├── all_results_summary.json")
    print(f"        ├── nounpp/nounpp_results.json")
    print(f"        ├── blimp/blimp_results.json")
    print(f"        └── repeat/repeat_surprisal_results.json")
    print("="*60)

except subprocess.CalledProcessError as e:
    print(f"❌ Error submitting grid search job: {e}")
    
    # Update metadata with error
    metadata["status"] = "failed"
    metadata["error"] = str(e)
    metadata["error_time"] = str(datetime.now())
    
    with open(exp_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    sys.exit(1)

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    
    # Update metadata with error
    metadata["status"] = "failed"
    metadata["error"] = str(e)
    metadata["error_time"] = str(datetime.now())
    
    with open(exp_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    sys.exit(1)

print(f"\n🚀 Experiment '{exp_name}' launched successfully!")
print(f"Use 'squeue -u $USER' to monitor job status")
print(f"Results will appear in: {results_dir}")