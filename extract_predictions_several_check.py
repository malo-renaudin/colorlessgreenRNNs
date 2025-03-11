import copy
from tqdm import tqdm
from torch.autograd import Variable
import argparse
import time
import os
import pandas as pd
import torch
import numpy as np
import h5py

from src.language_models.dictionary_corpus import Dictionary, tokenize
from src.language_models.model import RNNModel as lstm

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--checkpoints', nargs='+', required=True, help='List of model checkpoint paths')
parser.add_argument('-i', '--input', required=True, help='Input sentences in Tal\'s format')
parser.add_argument('-v', '--vocabulary', default='/scratch2/mrenaudin/colorlessgreenRNNs/english_data')
parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
parser.add_argument('--cuda', action='store_true', default=False)
args = parser.parse_args()

vocab = Dictionary(args.vocabulary)
sentences = [l.rstrip('\n').split(' ') for l in open(args.input + '.text', encoding='utf-8')]
gold = pd.read_csv(args.input + '.gold', sep='\t', header=None, names=['verb_pos', 'correct', 'wrong', 'nattr'])

results = []

with open("/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt", "r") as f:
    conditions = ["\t".join(line.strip().split("\t")[2:4]) for line in f]
    
checkpoint_dir = args.checkpoints[0]  # Assuming the first argument is the directory containing checkpoints
if os.path.isdir(checkpoint_dir):
    # Get all files in the directory, assuming they are PyTorch model checkpoints
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth') or f.endswith('.pt')]  # Adjust file extensions as needed
else:
    checkpoint_files = args.checkpoints  # If it's not a directory, treat it as a list of files

for checkpoint in checkpoint_files:
    print(f'Loading model: {checkpoint}')
    model = lstm('LSTM', 50001, 200, 650, 2, 0.2, False)
    with open(checkpoint, 'rb') as f:
        state_dict = torch.load(f, map_location='cuda' if args.cuda else 'cpu')
        model.load_state_dict(state_dict)
    model.eval()

    log_p_targets_correct = np.zeros((len(sentences), 1))
    log_p_targets_wrong = np.zeros((len(sentences), 1))
    #condition_accuracies = {condition[0] + " & " + condition[1]: [] for condition in conditions}  # Create a dictionary for condition accuracies
    condition_accuracies = {condition: [] for condition in conditions}

    for i, s in enumerate(tqdm(sentences)):
        hidden = model.init_hidden(1)
        for j, w in enumerate(s):
            if w not in vocab.word2idx:
                w = '<unk>'
            inp = Variable(torch.LongTensor([[vocab.word2idx[w]]]))
            if args.cuda:
                inp = inp.cuda()
            out, hidden = model(inp, hidden)
            out = torch.nn.functional.log_softmax(out[0]).unsqueeze(0)
            if j == gold.loc[i, 'verb_pos'] - 1:
                log_p_targets_correct[i] = out[0, 0, vocab.word2idx[gold.loc[i,'correct']]].item()
                log_p_targets_wrong[i] = out[0, 0, vocab.word2idx[gold.loc[i, 'wrong']]].item()
        #####Problème ici !!!!! attention à condition[i]       
        condition = conditions[i]
        correct = log_p_targets_correct[i] > log_p_targets_wrong[i]
        #condition_accuracies[condition[0]+ " & " + condition[1]].append(correct)
        condition_accuracies[condition].append(correct)

    condition_accuracy_results = {condition: np.mean(condition_accuracies[condition]) * 100 for condition in condition_accuracies}
    checkpoint_results = {'checkpoint': checkpoint}
    checkpoint_results.update(condition_accuracy_results)
    results.append(checkpoint_results)
    
results_df = pd.DataFrame(results)
results_df.to_csv(args.output, index=False)
print(f'Results saved to {args.output}')
