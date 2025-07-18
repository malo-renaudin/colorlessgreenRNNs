import sys
import os

sys.path.append('colorlessgreenRNNs')

from src.language_models import model as m
import torch
from evaluation_notebooks.utils import BLiMPDataset, collate_fn_blimp, NounPPDataset, collate_fn_nounpp
from wm_tests.utils import WMTestDataset, collate_fn, eval
from src.language_models.dictionary_corpus import Dictionary
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import json
import argparse
from wm_xp.tests.base_tests.blimp import eval_all_blimp
from wm_xp.tests.base_tests.nounpp import eval_nounpp
from wm_xp.tests.wm_tests.repeat_surprisal import eval_repeat_surprisal
from datetime import datetime



def main():
    parser = argparse.ArgumentParser(description='Test model checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--data_path', required=True, help = 'Path to the training data' )
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--nheads', type=int)
    parser.add_argument('--gumbel', action='store_true')
    parser.add_argument('--nounpp', type=str, default = 'colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt',
                       help = 'Path to NounPP dataset')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--cat_or_rand', type=str, default='cat', help='categorized or random test set for repeat surprisal')
    parser.add_argument('--sce', type=int, default=1, help='scenarion number for repeat sur^risal test set')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found")
        sys.exit(1)
        
    device = torch.device("cuda" if args.cuda else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    temperature = checkpoint.get('temperature', 1.0)
    

    
    all_results = {}
    
    try:
        # NounPP evaluation
        all_results['nounpp'] = eval_nounpp(
            checkpoint, 
            temperature, 
            args.hidden_dim, 
            args.nheads, 
            args.data_path, 
            args.nounpp, 
            device, 
        )
        
        # BLiMP evaluation
        all_results['blimp'] = eval_all_blimp(
            checkpoint, 
            args.data_path, 
            args.gumbel, 
            args.nheads, 
            temperature, 
            args.hidden_dim, 
            device,
        )
        
        # Repeat surprisal evaluation
        all_results['repeat'] = eval_repeat_surprisal(
            checkpoint, 
            args.data_path,
            args.cat_or_rand,
            args.sce,
            args.nheads,
            args.gumbel,
            args.hidden_dim,
            device,
        )
        
        # Save combined results summary
        with open(f"{args.output_dir}/all_results_summary.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("="*60)
        print("ALL EVALUATIONS COMPLETED SUCCESSFULLY!")
        print(f"Results saved to {args.output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Save error info
        error_info = {
            'checkpoint': args.checkpoint,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': str(datetime.now())
        }
        
        with open(f"{args.output_dir}/error.json", 'w') as f:
            json.dump(error_info, f, indent=2)
        
        sys.exit(1)

if __name__ == "__main__":
    main()