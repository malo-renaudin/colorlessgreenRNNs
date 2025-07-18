import sys
import os

sys.path.append('colorlessgreenRNNs')

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from src.language_models.dictionary_corpus import Dictionary
from collections import defaultdict
import torch.nn as nn
import src.language_models.model as m
import math
import torch.nn.functional as F
from wm_tests.utils import WMTestDataset, collate_fn, eval
from pathlib import Path
from tqdm import tqdm

def create_dataloader(base_path, file_suffix, batch_size, dictionary):

    data_file = f'{base_path}/{file_suffix}.txt'
    marker_file = f'{base_path}/{file_suffix}_markers.txt'
    dataset = WMTestDataset(data_file, marker_file, dictionary)
    return DataLoader(dataset, batch_size, collate_fn=collate_fn)

def load_test(cat_or_rand, sce, batch_size, dictionary, test_types):

    base_path = 'colorlessgreenRNNs/wm_tests/rnn_input_files'
        
    
    prefix_map = {
        'cat': f'categorized_lists_sce{sce}',
        'rand': f'random_lists_sce{sce}'
    }
    
    if cat_or_rand not in prefix_map:
        raise ValueError(f"cat_or_rand must be 'cat' or 'rand', got: {cat_or_rand}")
    
    prefix = prefix_map[cat_or_rand]
    
    dataloaders = {
        f'{cat_or_rand}_s{sce}_{test_type}_dataloader': create_dataloader(base_path, f'{prefix}_{test_type}',230, dictionary)
        for test_type in test_types
    }
    
    return dataloaders
         
def eval_repeat_surprisal(checkpoint, 
         data_path,
         cat_or_rand,
         sce,
         nheads,
         gumbel,
         hidden_dim,
         device,
         batch_size=230
         ):
    
    res = {}

    data_path = "colorlessgreenRNNs/english_data"
    dictionary = Dictionary(data_path)

    model = m.CBR_RNN(50001, hidden_dim, hidden_dim, nheads, 0.5, device)
    model=model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_types = ['control', 'permute', 'repeat']

    dataloaders = load_test(cat_or_rand, sce, batch_size, dictionary, test_types)
    temperature = checkpoint['temperature']
    for type in test_types:

        dataloader = dataloaders[f'{cat_or_rand}_s{sce}_{type}_dataloader']
        res[type]= eval(model, dataloader, nheads, temperature, gumbel,device)

    return res 

