#!/usr/bin/env python3
"""
Simple experiment runner script.
"""
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
from xp_argparser import xp_parser
from grid_search.configs.generate_config_file import generate_config
import torch
from base_tests.nounpp import eval as eval_nounpp
from parallel_evaluation import run_evaluation

#call arguments
parser = argparse.ArgumentParser(
    parents=[xp_parser], description="Grid Search over CBR-RNN hyperparameters and evaluation of each configuration"
)
args = parser.parse_args()
device = torch.device("cuda" if args.cuda else "cpu")

# Create experiment folder structure
exp_name = args.name
exp_dir = Path(f"wm_xp/experiments/{exp_name}")
config_dir = exp_dir / "config"
checkpoints_dir = exp_dir / "checkpoints"

# Create all directories
exp_dir.mkdir(parents=True, exist_ok=True)
config_dir.mkdir(exist_ok=True)
checkpoints_dir.mkdir(exist_ok=True)


# Create config file in the config subfolder
config_file_path = config_dir / args.name 
print("Creating config...")
generate_config(args.hidden_dims, args.num_heads, args.temperatures, args.gumbel_softmax, str(config_file_path))

# Run grid search
print("Running grid search...")
subprocess.run(["bash", args.grid_search_script, str(config_file_path), str(checkpoints_dir)], check=True)

#Evaluate
#On NounPP
run_evaluation(checkpoints_dir,args.data_path, args.nounpp, eval_nounpp, device)