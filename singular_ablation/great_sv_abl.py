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
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


##########################################################################################
# General Parameters and Necessary Files 
##########################################################################################
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"
dictionary = Dictionary(data_path)
nounpp = "//scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt"
layers = [0,1]
weight_type = 'hh'
gates = ['cell','forget','input','output']
ntokens = 50001


nb_abl_per_weight = 3


##########################################################################################
# NounPP Dataset Class
##########################################################################################

class NounPPDataset(Dataset):
    def __init__(self, nounpp_file, dictionary):
        self.sentences = []
        self.conditions = []
        self.correct = []
        self.wrong = []
        self.encoded_sentences = []
        self.encoded_correct = []
        self.encoded_wrong = []
        self.dictionary = dictionary

        with open(nounpp_file, "r") as f:
            for line in f:
                line = line.split()
                sentence = line[1:7]
                condition = " ".join(line[7:9])
                wrong = line[9]
                correct = line[6]
                encoded_sentence = [
                    self.dictionary.word2idx.get(
                        word, self.dictionary.word2idx.get("<unk>")
                    )
                    for word in sentence
                ]
                encoded_correct = self.dictionary.word2idx.get(
                    correct, self.dictionary.word2idx.get("<unk>")
                )
                encoded_wrong = self.dictionary.word2idx.get(
                    wrong, self.dictionary.word2idx.get("<unk>")
                )

                self.sentences.append(sentence)
                self.conditions.append(condition)
                self.correct.append(correct)
                self.wrong.append(wrong)
                self.encoded_sentences.append(encoded_sentence)
                self.encoded_correct.append(encoded_correct)
                self.encoded_wrong.append(encoded_wrong)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "encoded_sentence": torch.tensor(
                self.encoded_sentences[idx], dtype=torch.long
            ),
            "correct": self.correct[idx],
            "encoded_correct": torch.tensor(
                self.encoded_correct[idx], dtype=torch.long
            ),
            "wrong": self.wrong[idx],
            "encoded_wrong": torch.tensor(self.encoded_wrong[idx], dtype=torch.long),
            "condition": self.conditions[idx],
        }

##########################################################################################
# Collate Function
##########################################################################################

def collate_fn(batch):
    """Custom collate function to properly handle sentences as lists of strings."""
    sentences = [item["sentence"] for item in batch]  # Keep lists of words as they are
    encoded_sentences = torch.stack(
        [item["encoded_sentence"] for item in batch]
    )  # Stack tensors
    encoded_correct = torch.stack([item["encoded_correct"] for item in batch])
    encoded_wrong = torch.stack([item["encoded_wrong"] for item in batch])
    correct = [item["correct"] for item in batch]
    wrong = [item["wrong"] for item in batch]
    conditions = [item["condition"] for item in batch]

    return {
        "sentence": sentences,
        "encoded_sentence": encoded_sentences,
        "correct": correct,
        "encoded_correct": encoded_correct,
        "wrong": wrong,
        "encoded_wrong": encoded_wrong,
        "condition": conditions,
    }


##########################################################################################
# Priming
##########################################################################################

init_sentence = " ".join(
    [
        "In service , the aircraft was operated by a crew of five and could accommodate either 30 paratroopers , 32 <unk> and 28 sitting casualties , or 50 fully equipped troops . <eos>",
        'He even speculated that technical classes might some day be held " for the better training of workmen in their several crafts and industries . <eos>',
        "After the War of the Holy League in 1537 against the Ottoman Empire , a truce between Venice and the Ottomans was created in 1539 . <eos>",
        'Moore says : " Tony and I had a good <unk> and off-screen relationship , we are two very different people , but we did share a sense of humour " . <eos>',
        "<unk> is also the basis for online games sold through licensed lotteries . <eos>",
    ]
)


def feed_input(model, hidden, w):
    inp = torch.autograd.Variable(
        torch.LongTensor([[dictionary.word2idx[w]]]).to(device)
    )
    out, hidden= model(inp, hidden)
    return out, hidden


def feed_sentence(model, h, sentence):
    outs = []
    for w in sentence:
        out, h = feed_input(model, h, w)
        outs.append(torch.nn.functional.log_softmax(out[0]).unsqueeze(0))
    return outs, h 


##########################################################################################
# Evaluation function
##########################################################################################


def eval(model, test_dataloader, init_sentence):
    condition_accuracies = defaultdict(int)
    condition_counts = defaultdict(int)
    correct_pred = 0
    sentence_details = []

    model.eval()

    hidden = move_to_device(model.init_hidden(1), device)
    init_out, init_h= feed_sentence(model, hidden, init_sentence.split(" "))
    
    with torch.no_grad():
        for batch in test_dataloader:
            out = None
            written = batch["sentence"]
            sentence = batch["encoded_sentence"].to(device)
            correct = batch["encoded_correct"].to(device)
            wrong = batch["encoded_wrong"].to(device)
            condition = batch["condition"]#.to(device)
            batch_size = sentence.size(0)
            hidden = (
                init_h[0].expand(-1, batch_size, -1).contiguous(),
                init_h[1].expand(-1, batch_size, -1).contiguous(),
            )
            # stack = (
            #     init_stack.expand(batch_size, -1, -1).contiguous(),
            # )
            # stack=stack[0]
            for w in range(sentence.shape[1] - 1):

                word = torch.autograd.Variable(sentence[:, w].unsqueeze(0))
                out, hidden = model(word, hidden)
                
            hidden = repackage_hidden(hidden)

            log_probs = torch.nn.functional.log_softmax(out, dim=-1)
            correct_log_probs = log_probs[0, torch.arange(batch_size), correct]
            wrong_log_probs = log_probs[0, torch.arange(batch_size), wrong]
            correct_predictions = correct_log_probs >= wrong_log_probs
            for i in range(batch_size):
                cond = condition[i]
                pred = correct_predictions[i].item()
                condition_counts[cond] += 1
                condition_accuracies[cond] += pred

                # sentence_details.append(
                #     {
                #         "sentence": written[i],
                #         "condition": condition[i],
                #         "correct_log_prob": correct_log_probs[i],
                #         "wrong_log_prob": wrong_log_probs[i],
                #         "model_prefers_correct": pred,
                #     }
                # )
    final_accuracies = {
        cond: condition_accuracies[cond] / condition_counts[cond]
        for cond in condition_accuracies
    }

    return final_accuracies


##########################################################################################
# Ablating Checkpoints SVs
##########################################################################################


def zero_out_singular_values(weight_matrix: torch.Tensor, n: int) -> torch.Tensor:
    # Compute SVD
    U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
    
    # Zero out the n smallest singular values
    S[n]=0
    
    # Reconstruct the matrix
    modified_weight = (U * S.unsqueeze(0)) @ Vh
    return modified_weight


def modify_checkpoint_weight(checkpoint_path, layer, weight_type, gate, n, device):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_weights = checkpoint["model_state_dict"]
    
    # Extract original weight tensor (concatenated gates)
    original_weight = model_weights[f'rnn.weight_{weight_type}_l{layer}']  # shape: (4*gate_dim, dim)
    
    # Split into gates
    gates = dict(zip(['input', 'forget', 'cell', 'output'], original_weight.chunk(4, dim=0)))
    
    # Modify specified gate's weight matrix
    modified_gate_weight = zero_out_singular_values(gates[gate], n)
    
    # Replace gate weight in gates dict
    gates[gate] = modified_gate_weight
    
    # Re-concatenate gates into full weight matrix
    modified_weight = torch.cat([gates[g] for g in ['input', 'forget', 'cell', 'output']], dim=0)
    
    # Create a copy of checkpoint to avoid modifying the original
    new_checkpoint = copy.deepcopy(checkpoint)
    new_checkpoint["model_state_dict"][f'rnn.weight_{weight_type}_l{layer}'] = modified_weight
    
    return new_checkpoint

def evaluate_on_n_ablations(checkpoint_path, ntokens, layers, weight_type, gates, test_dataloader, n, init_sentence=init_sentence):
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
                
            ppl_original = eval(model, test_dataloader, init_sentence)
            
            ablation[f'layer_{layer}'][gate]['original']=ppl_original
            ablation[f'layer_{layer}'][gate]['ablations'] = []
            
            for i in tqdm(range(n), desc=f"Layer {layer}, Gate {gate}"):
                check = modify_checkpoint_weight(checkpoint_path, layer, weight_type, gate, i, device)
                model = lstm('LSTM', ntokens, 650, 650, 2, 0, False).to(device)
                model.load_state_dict(check['model_state_dict'])
                ppl = eval(model, test_dataloader, init_sentence)
                ablation[f'layer_{layer}'][gate]['ablations'].append(ppl)
                
    return ablation   

##########################################################################################
# Main Loop
##########################################################################################

test_dataset = NounPPDataset(nounpp, dictionary)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

checkpoint_dir_str = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/lstm_adam_full_check_shuffled"
checkpoint_files =  [f'epoch_1_batch_{i}.pt' for i in range(0, 301, 1)] + [f'epoch_1_batch_{i}.pt'for i in range(400, 9300, 100)]+[f'epoch_{i}.pt' for i in range(1, 41, 1)]
checkpoint_dir = Path(checkpoint_dir_str)

ablation = {}
for item_name in checkpoint_files:
    item_path = checkpoint_dir / item_name
    ablation[f'{item_path}'] = evaluate_on_n_ablations(item_path, ntokens, layers, weight_type, gates, test_dataloader, nb_abl_per_weight)

torch.save(ablation, '/scratch2/mrenaudin/colorlessgreenRNNs/singular_ablation/great_sv_ablations.pt')