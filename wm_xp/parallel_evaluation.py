#!/usr/bin/env python3
"""
Simple parallel evaluation
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json

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

def run_evaluation(checkpoints_dir, data_path, eval_dataset, eval_func, device):
    """Run evaluation on all configs in parallel"""
    config_dirs = [d for d in Path(checkpoints_dir).iterdir() if d.is_dir()]
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(eval_single_config, config_dir, data_path, eval_dataset, eval_func, device) 
            for config_dir in config_dirs
        ]
        for future in futures:
            print(future.result())

