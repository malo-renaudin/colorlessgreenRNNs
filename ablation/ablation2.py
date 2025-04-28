import sys
import os
import pickle
from tqdm import tqdm

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level from the script's directory (evaluation_notebooks)
parent_dir = os.path.dirname(script_dir)
# Join with 'src' to get the correct path to the src directory
src_dir = os.path.join(parent_dir, "src")
sys.path.append(os.path.abspath(src_dir))
import torch
from torch.utils.data import Dataset, DataLoader
from language_models.dictionary_corpus import Dictionary, Corpus, tokenize
from language_models.model import RNNModel as lstm
from language_models.utils import move_to_device
import random
import pandas as pd
from collections import defaultdict
import numpy as np
import argparse
import re
from utils import NounPPDataset, collate_fn, evaluate_checkpoint, feed_input, feed_sentence, eval, ablate_neuron, restore_neuron, cache_weights


batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"
dictionary = Dictionary(data_path)
nounpp = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt"
checkpoint_dir = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/lstm_sgd_lr10"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_name, device):
    model = lstm("LSTM", 50001, 650, 650, 2, 0.2, False).to(device)
    with open(checkpoint_name, "rb") as f:
        state_dict = torch.load(f, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
    return model

init_sentence = " ".join(
    [
        "In service , the aircraft was operated by a crew of five and could accommodate either 30 paratroopers , 32 <unk> and 28 sitting casualties , or 50 fully equipped troops . <eos>",
        'He even speculated that technical classes might some day be held " for the better training of workmen in their several crafts and industries . <eos>',
        "After the War of the Holy League in 1537 against the Ottoman Empire , a truce between Venice and the Ottomans was created in 1539 . <eos>",
        'Moore says : " Tony and I had a good <unk> and off-screen relationship , we are two very different people , but we did share a sense of humour " . <eos>',
        "<unk> is also the basis for online games sold through licensed lotteries . <eos>",
    ]
)

test_dataset = NounPPDataset(nounpp, dictionary)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

checkpoint_files = [
    f
    for f in os.listdir(checkpoint_dir)
    if f.startswith("epoch_") and f.endswith(".pt")
]

results = {}

for checkpoint_file in tqdm(checkpoint_files, desc="Evaluating checkpoints"):

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    checkpoint_name = checkpoint_file
    model = load_model(checkpoint_path, device)
    res = evaluate_checkpoint(model, test_dataloader, init_sentence, dictionary, device)
    results[checkpoint_name] = res



with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)