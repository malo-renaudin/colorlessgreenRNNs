import sys
import os

sys.path.append('/scratch2/mrenaudin/colorlessgreenRNNs')

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from src.language_models.dictionary_corpus import Dictionary
from collections import defaultdict
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
import src.language_models.model as m
import math
import torch.nn.functional as F
from utils import WMTestDataset, collate_fn, eval
from pathlib import Path
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 230 #sentences change length every 230 sentences
model = m.CBR_RNN(50001, 650, 650, 1, 0, device)
data_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"
dictionary = Dictionary(data_path)
checkpoint_dir_str = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/test_attention_650"
checkpoint_files =  [f'epoch_{i}.pt' for i in range(1, 39, 1)]
checkpoint_dir = Path(checkpoint_dir_str)
types = ['control', 'permute', 'repeat']

#Cat s3 control
cat_s3_control_marker = '/scratch2/mrenaudin/colorlessgreenRNNs/wm_tests/rnn_input_files/categorized_lists_sce3_control_markers.txt'
cat_s3_control = '/scratch2/mrenaudin/colorlessgreenRNNs/wm_tests/rnn_input_files/categorized_lists_sce3_control.txt'
cat_s3_control_dataset = WMTestDataset(cat_s3_control,cat_s3_control_marker, dictionary)
cat_s3_control_dataloader = DataLoader(cat_s3_control_dataset, batch_size=batch_size, collate_fn=collate_fn)
#Cat s3 permute
cat_s3_permute_marker = '/scratch2/mrenaudin/colorlessgreenRNNs/wm_tests/rnn_input_files/categorized_lists_sce3_permute_markers.txt'
cat_s3_permute = '/scratch2/mrenaudin/colorlessgreenRNNs/wm_tests/rnn_input_files/categorized_lists_sce3_permute.txt'
cat_s3_permute_dataset = WMTestDataset(cat_s3_permute,cat_s3_permute_marker, dictionary)
cat_s3_permute_dataloader = DataLoader(cat_s3_permute_dataset, batch_size=batch_size, collate_fn=collate_fn)

#Cat s3 repeat
cat_s3_repeat_marker = '/scratch2/mrenaudin/colorlessgreenRNNs/wm_tests/rnn_input_files/categorized_lists_sce3_repeat_markers.txt'
cat_s3_repeat = '/scratch2/mrenaudin/colorlessgreenRNNs/wm_tests/rnn_input_files/categorized_lists_sce3_repeat.txt'
cat_s3_repeat_dataset = WMTestDataset(cat_s3_repeat,cat_s3_repeat_marker, dictionary)
cat_s3_repeat_dataloader = DataLoader(cat_s3_repeat_dataset, batch_size=batch_size, collate_fn=collate_fn)

dataloaders = {
    'control': cat_s3_control_dataloader,
    'permute': cat_s3_permute_dataloader,
    'repeat': cat_s3_repeat_dataloader
}

eval_tt = {}
for item_name in tqdm(checkpoint_files, desc="Processing checkpoints"):
    item_path = checkpoint_dir / item_name
    checkpoint = torch.load(item_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    temperature = checkpoint['epoch'] #error in checkpointing during training
    epoch = checkpoint['temperature']
    eval_tt[epoch]={}
    for type in types:
        dataloader = dataloaders[type]
        eval_tt[epoch][type]= eval(model, dataloader, nheads=1, temperature=temperature, gs=True)

torch.save(eval_tt, '/scratch2/mrenaudin/colorlessgreenRNNs/wm_tests/results/test_attention_650_sce3')