#!/usr/bin/env python3
"""
Simple parallel evaluation
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import torch
import os

def get_optimal_workers():
    """Auto-detect optimal number of workers"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        # For evaluation, we can often handle more workers than GPUs
        # since models are smaller during inference
        optimal = min(gpu_count * 2, 8)  # 2x GPUs but cap at 8
        print(f"Detected {gpu_count} GPUs, using {optimal} workers")
        return optimal
    else:
        cpu_count = os.cpu_count()
        optimal = min(cpu_count - 1, 4)  # Leave 1 core free
        print(f"No GPU detected, using {optimal} CPU workers")
        return optimal

def eval_single_config(config_dir, data_path, eval_dataset, eval_func, device):
    """Evaluate all checkpoints in one config directory"""
    # Parse config from directory name: h1024_heads1_t0.1_true
    parts = config_dir.name.split('_')
    hidden_dim = int(parts[0][1:])
    nheads = int(parts[1][5:])
    temperature = float(parts[2][1:])
    
    # Call your eval function
    results = eval_func(temperature, hidden_dim, nheads, data_path, eval_dataset, device)
    
    # Save results
    output_file = config_dir.parent.parent / "results" / f"{eval_dataset}"/f"{config_dir.name}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f)
    
    return f"Completed {config_dir.name}"

def run_evaluation(checkpoints_dir, data_path, eval_dataset, eval_func, max_workers,device):
    """Run evaluation on all configs in parallel"""
    config_dirs = [d for d in Path(checkpoints_dir).iterdir() if d.is_dir()]
    nb_workers = get_optimal_workers()
    with ProcessPoolExecutor(max_workers=nb_workers) as executor:
        futures = [
            executor.submit(eval_single_config, config_dir, data_path, eval_dataset, eval_func, device) 
            for config_dir in config_dirs
        ]
        for future in futures:
            print(future.result())

