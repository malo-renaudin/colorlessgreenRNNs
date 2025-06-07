import sys
import os

sys.path.append('/scratch2/mrenaudin/colorlessgreenRNNs')

from src.language_models.model import RNNModel as lstm
import torch
from wm_tests.utils import WMTestDataset, collate_fn
from src.language_models.dictionary_corpus import Dictionary
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.language_models.utils import move_to_device, repackage_hidden
from utils import evaluate_checkpoint
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"
dictionary = Dictionary(data_path)
model = lstm("LSTM", len(dictionary), 650, 650, 2, 0.2, False).to(device)
checkpoint = '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/lstm_adam_full_check_shuffled/epoch_40.pt'
batch_size=230

repeat_marker = '/scratch2/mrenaudin/colorlessgreenRNNs/wm_tests/rnn_input_files/categorized_lists_sce3_repeat_markers.txt'
repeat = '/scratch2/mrenaudin/colorlessgreenRNNs/wm_tests/rnn_input_files/categorized_lists_sce3_repeat.txt'
repeat_dataset = WMTestDataset(repeat,repeat_marker, dictionary)
repeat_dataloader = DataLoader(repeat_dataset, batch_size=batch_size, collate_fn=collate_fn)

################################################################################
#Evaluation function
################################################################################


def eval(model, dataloader, batch_size):
    all_repeat_surprisals = {}
    model.eval()
    hidden = move_to_device(model.init_hidden(batch_size), device)
    # Forward pass with hidden state update word by word
    with torch.no_grad():
        for batch in dataloader:
            encoded_sentence = batch["encoded_sentence"]
            condition = batch["condition"]
            marker = batch["marker"]
            batch_size, seq_len = encoded_sentence.shape
            input_seq = encoded_sentence[:, :-1].transpose(0, 1).to(device)  # (seq_len-1, batch_size)
            target_seq = encoded_sentence[:, 1:].transpose(0, 1).to(device)
            
            
            output, hidden = model(input_seq,hidden)
            hidden = repackage_hidden(hidden)
            
            log_probs = F.log_softmax(output, dim=-1)  #
            
            nll_loss = F.nll_loss(
                        log_probs.reshape(-1, log_probs.size(-1)),  # ( (seq_len-1)*batch_size, vocab_size )
                        target_seq.reshape(-1),                         # ((seq_len-1)*batch_size)
                        reduction='none'
                    )
                    
                    # Reshape back to (seq_len-1, batch_size)
            nll_loss = nll_loss.view(seq_len - 1, batch_size).transpose(0, 1)  # (batch_size, seq_len-1)        
            mask_list1 = (marker[:, 1:] == 1)  # remove first token since nll_loss aligns with shifted target
            mask_list2 = (marker[:, 1:] == 3)
            
            # Extract surprisal for each list and reshape
            # Number of tokens in each list should be condition[0][0]*2 (including punctuation)
            list_len = condition[0][0] * 2
            
            surprisal_list1 = nll_loss[mask_list1].view(batch_size, list_len)
            surprisal_list2 = nll_loss[mask_list2].view(batch_size, list_len)
            
            # Select repeated word indices (odd positions assuming repeats are at odd indices)
            word_indices = torch.arange(0, condition[0][0]*2, step=2)  # e.g., 1, 3, 5, ...
            word_indices = word_indices[1:]#get rid of first word of the list
  
            surprisal1_repeats = surprisal_list1[:, word_indices]
            surprisal2_repeats = surprisal_list2[:, word_indices]
            
            # Compute repeat surprisal ratio as percentage
            repeat_surprisal = (surprisal2_repeats / surprisal1_repeats) * 100
            all_repeat_surprisals[f'list len : {condition[0][0]}, prompt len : {condition[0][1]}']=repeat_surprisal
            
    return all_repeat_surprisals

################################################################################
#Ablation functions
################################################################################
def ablate_neuron(model, l, n, num_neurons):
    gate_indices = torch.tensor(
                [
                    n,
                    n + num_neurons,
                    n + num_neurons * 2,
                    n + num_neurons * 3,
                ]
            )
    if l == 0:
        # Ablate layer 0 neuron
        model.rnn.weight_ih_l0[gate_indices] = 0
        model.rnn.weight_hh_l0[gate_indices] = 0
        model.rnn.bias_ih_l0[gate_indices] = 0
        model.rnn.bias_hh_l0[gate_indices] = 0
    elif l == 1:
        # Ablate layer 1 neuron
        model.rnn.weight_ih_l1[gate_indices] = 0
        model.rnn.weight_hh_l1[gate_indices] = 0
        model.rnn.bias_ih_l1[gate_indices] = 0
        model.rnn.bias_hh_l1[gate_indices] = 0
    return model, gate_indices

def restore_neuron(model, l, gate_indices, weights):
    if l == 0:
        model.rnn.weight_ih_l0[gate_indices] = weights["weight_ih_l0"][gate_indices]    
        model.rnn.weight_hh_l0[gate_indices] = weights["weight_hh_l0"][gate_indices]
        model.rnn.bias_ih_l0[gate_indices] = weights["bias_ih_l0"][gate_indices]
        model.rnn.bias_hh_l0[gate_indices] = weights["bias_hh_l0"][gate_indices]
    elif l == 1:
        model.rnn.weight_ih_l1[gate_indices] = weights["weight_ih_l1"][gate_indices]
        model.rnn.weight_hh_l1[gate_indices] = weights["weight_hh_l1"][gate_indices]
        model.rnn.bias_ih_l1[gate_indices] = weights["bias_ih_l1"][gate_indices]
        model.rnn.bias_hh_l1[gate_indices] = weights["bias_hh_l1"][gate_indices]
    return model

def cache_weights(model):
    weights = {
    "weight_ih_l0": model.rnn.weight_ih_l0.clone(),
    "weight_hh_l0": model.rnn.weight_hh_l0.clone(),
    "bias_ih_l0": model.rnn.bias_ih_l0.clone(),
    "bias_hh_l0": model.rnn.bias_hh_l0.clone(),
    "weight_ih_l1": model.rnn.weight_ih_l1.clone(),
    "weight_hh_l1": model.rnn.weight_hh_l1.clone(),
    "bias_ih_l1": model.rnn.bias_ih_l1.clone(),
    "bias_hh_l1": model.rnn.bias_hh_l1.clone(),
}
    return weights

def evaluate_checkpoint(model, test_loader, batch_size, layer=2, num_neurons=650):
    results = {}
    original_accuracies = eval(model, test_loader, batch_size)
    results["original"] = original_accuracies

    ####caching
    weights= cache_weights(model)

    for l in range(layer):
        for n in tqdm(range(num_neurons), desc="Ablating neurons"):
            with torch.no_grad():
                
                model, gate_indices = ablate_neuron(model, l, n, num_neurons)
                res = eval(model, test_loader, batch_size)
                results[f"layer_{l}_neuron_{n}"] = res
                #restore weights
                model = restore_neuron(model, l, gate_indices, weights)
    return results
        
res = evaluate_checkpoint(model, repeat_dataloader, batch_size)
torch.save(res, '/scratch2/mrenaudin/colorlessgreenRNNs/ablation/wm_abl')