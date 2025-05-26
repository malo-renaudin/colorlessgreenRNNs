import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level from the script's directory (evaluation_notebooks)
parent_dir = os.path.dirname(script_dir)
# Join with 'src' to get the correct path to the src directory
src_dir = os.path.join(parent_dir, "src")
sys.path.append(os.path.abspath(src_dir))

import torch
from language_models.dictionary_corpus import Dictionary, Corpus, tokenize
from language_models.model import RNNModel as lstm
from language_models.utils import move_to_device, batchify, get_batch, repackage_hidden
import torch.nn as nn
import math
from pathlib import Path
import copy
from tqdm import tqdm

device = torch.device("cuda")
eval_batch_size = 10
corpus = Corpus('/scratch2/mrenaudin/colorlessgreenRNNs/english_data')
ntokens = len(corpus.dictionary)
val_data = batchify(corpus.valid, eval_batch_size, device)
layers = [0,1]
weight_type = 'hh'
gates = ['cell','forget','input','output']

################################################################################################################
# Evaluation function
################################################################################################################

def evaluate(model, val_data, ntokens):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = move_to_device(model.init_hidden(eval_batch_size), device)
    with torch.no_grad():
        for i in range(0, val_data.size(0) - 1, 35):
            data, targets = get_batch(val_data, i, 35)
            data, targets = data.to(device), targets.to(device)
            
            
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += (
                len(data) * nn.CrossEntropyLoss()(output_flat, targets).item()
            )
            del output, output_flat
            hidden = repackage_hidden(hidden)
            
    loss = total_loss / (len(val_data) - 1)
    
    return math.exp(loss)


################################################################################################################
# Modify checkpoint
################################################################################################################


def zero_out_lowest_singular_values(weight_matrix: torch.Tensor, n: int) -> torch.Tensor:
    # Compute SVD
    U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
    
    # Zero out the n smallest singular values
    if n > 0:
        S[-n:] = 0
    
    # Reconstruct the matrix
    modified_weight = (U * S.unsqueeze(0)) @ Vh
    return modified_weight

def modify_checkpoint_weight(checkpoint_path, layer, weight_type, gate, n_singular_values_to_zero, device):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_weights = checkpoint["model_state_dict"]
    
    # Extract original weight tensor (concatenated gates)
    original_weight = model_weights[f'rnn.weight_{weight_type}_l{layer}']  # shape: (4*gate_dim, dim)
    
    # Split into gates
    gates = dict(zip(['input', 'forget', 'cell', 'output'], original_weight.chunk(4, dim=0)))
    
    # Modify specified gate's weight matrix
    modified_gate_weight = zero_out_lowest_singular_values(gates[gate], n_singular_values_to_zero)
    
    # Replace gate weight in gates dict
    gates[gate] = modified_gate_weight
    
    # Re-concatenate gates into full weight matrix
    modified_weight = torch.cat([gates[g] for g in ['input', 'forget', 'cell', 'output']], dim=0)
    
    # Create a copy of checkpoint to avoid modifying the original
    new_checkpoint = copy.deepcopy(checkpoint)
    new_checkpoint["model_state_dict"][f'rnn.weight_{weight_type}_l{layer}'] = modified_weight
    
    return new_checkpoint

################################################################################################################
# Evaluate 1 checkpoint on n ablations
################################################################################################################

def evaluate_on_n_ablations(checkpoint_path, ntokens, layers, weight_type, gates, n):
    ablation ={}
    for layer in layers:
        ablation[f'layer_{layer}'] = {}
        for gate in gates:
            
            ablation[f'layer_{layer}'][gate] = {}
            
            model = lstm('LSTM', ntokens, 650, 650, 2, 0, False).to(device)
            with open(checkpoint_path, "rb") as f:
                state_dict = torch.load(
                    f, map_location="cuda" if device == "cuda" else "cpu"
                )
                model.load_state_dict(state_dict["model_state_dict"])
                
            ppl_original = evaluate(model, val_data, ntokens)
            
            ablation[f'layer_{layer}'][gate]['original']=ppl_original
            ablation[f'layer_{layer}'][gate]['ablations'] = []
            
            for i in tqdm(range(n), desc=f"Layer {layer}, Gate {gate}"):
                check = modify_checkpoint_weight(checkpoint_path, layer, weight_type, gate, i+1, device)
                model = lstm('LSTM', ntokens, 650, 650, 2, 0, False).to(device)
                model.load_state_dict(check['model_state_dict'])
                ppl = evaluate(model, val_data, ntokens)
                ablation[f'layer_{layer}'][gate]['ablations'].append(ppl)
                
    return ablation   


################################################################################################################
# Main : loop over checkpoints
################################################################################################################

checkpoint_dir_str = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/lstm_adam_full_check_shuffled"
checkpoint_files =  [f'epoch_1_batch_{i}.pt' for i in range(0, 301, 1)] + [f'epoch_1_batch_{i}.pt'for i in range(400, 9300, 100)]+[f'epoch_{i}.pt' for i in range(1, 41, 1)]
checkpoint_dir = Path(checkpoint_dir_str)

ablation = {}
for item_name in checkpoint_files:
    item_path = checkpoint_dir / item_name
    ablation[f'{item_path}'] = evaluate_on_n_ablations(item_path, ntokens, layers, weight_type, gates, 100)

torch.save(ablation, '/scratch2/mrenaudin/colorlessgreenRNNs/singular_ablation/ablations.pt')