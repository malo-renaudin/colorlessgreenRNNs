import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.append(os.path.abspath(src_dir))

import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from language_models.dictionary_corpus import Dictionary, Corpus, tokenize
from language_models.model import RNNModel as lstm
from language_models.model import CBR_RNN as cbr
from language_models.utils import move_to_device
from torch.nn.functional import scaled_dot_product_attention
import torch.nn as nn
import random
import pandas as pd
from collections import defaultdict
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser(description="Evaluation of LSTM on NounPP")
parser.add_argument("--emsize", type=int, help="size of word embeddings")
parser.add_argument("--nhid", type=int, help="size of hidden state")
parser.add_argument("--nheads", type=int, help="number of attention heads")
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="directory containing checkpoints for training the model",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/scratch2/mrenaudin/colorlessgreenRNNs/evaluation_notebooks/results",
    help="directory to save the results dataframe (default: /scratch2/mrenaudin/colorlessgreenRNNs/evaluation_notebooks/results)",
)
parser.add_argument(
    "--output_name",
    type=str,
    help="name of the results dataframe file (default: evaluation_results.csv)",
)
args = parser.parse_args()


batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"
dictionary = Dictionary(data_path)
nounpp = "//scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt"
checkpoint_dir = args.checkpoint_dir
output_dir = args.output_dir
output_name = args.output_name


# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_name)


class CBR_RNN(nn.Module):
    # goal here is to reuse CBR_RNN but with scaled dot product attention for more efficient computations.
    # Also I got rid of options such as loading pretrained embeddings, and ablating attention to simplify the code.
    # In the future if those options are needed, they can still be copy pasted from William's code as the structure hasn't changed
    def __init__(self, ntoken, ninp, nhid, nheads, dropout=0.5, device=None):
        super().__init__()
        # same layers as Timkey
        self.device = device
        self.nheads = nheads
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.score_attn = nn.Softmax(dim=-1)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.q = nn.Linear(ninp + nhid, nhid)
        self.intermediate_h = nn.Linear(nhid * 4, nhid * 4)
        self.decoder = nn.Linear(nhid, ntoken)
        self.q_norm = torch.nn.LayerNorm(nhid)
        self.int_norm = torch.nn.LayerNorm(nhid * 4)
        self.f_norm = torch.nn.LayerNorm(nhid * 3)
        self.nhid = nhid
        self.final_h = nn.Linear(nhid * 4, nhid * 3)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=nhid, num_heads=nheads, batch_first=True
        )

        self.init_weights()

    def init_weights(self):
        """Initialize model weights for better training dynamics"""
        # General initialization
        for name, param in self.named_parameters():
            if "weight" in name:
                if "norm" in name:
                    nn.init.ones_(param)
                elif "encoder" in name:
                    nn.init.normal_(param, mean=0, std=0.01)
                elif "decoder" in name:
                    nn.init.normal_(param, mean=0, std=0.01)
                else:
                    # Standard He initialization for processing layers
                    nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="tanh")
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_cache(self, observation, nheads):
        """Initialize hidden state and attention caches with better initialization strategy"""
        if len(observation.size()) > 1:
            bsz = observation.size(dim=-1)
        else:
            bsz = 1

        hidden = torch.zeros(1, bsz, self.nhid).to(self.device) 
        if nheads == 1:
            key_cache = torch.zeros(bsz, 1, 1, self.nhid).to(self.device) 
            value_cache = torch.zeros(bsz, 1, 1, self.nhid).to(self.device) 
        else:
            key_cache = torch.zeros(bsz, 1, self.nhid).to(self.device) 
            value_cache = torch.zeros(bsz, 1, self.nhid).to(self.device) 
        return hidden, key_cache, value_cache


    def update_cache(self, key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i, nheads):
        hidden_i = hidden_i.unsqueeze(0)
        hidden = torch.cat((hidden, hidden_i), dim=0)
        if nheads == 1:
                key_cache_i = key_cache_i.unsqueeze(1).unsqueeze(1)
                value_cache_i = value_cache_i.unsqueeze(1).unsqueeze(1)
                key_cache = torch.cat((key_cache, key_cache_i), dim=2)
                value_cache = torch.cat((value_cache, value_cache_i), dim=2)
        else:
            key_cache_i = key_cache_i.unsqueeze(1)
            value_cache_i = value_cache_i.unsqueeze(1)
            key_cache = torch.cat((key_cache, key_cache_i), dim=1)
            value_cache = torch.cat((value_cache, value_cache_i), dim=1)
            
        return key_cache, value_cache, hidden
    
    
    def attention_layer(self, query, key_cache, value_cache, nheads):
        if nheads == 1:
                query = query.unsqueeze(1)
                
                # Ensure all tensors are on the same device
                if query.device != key_cache.device:
                    key_cache = key_cache.to(query.device)
                if query.device != value_cache.device:
                    value_cache = value_cache.to(query.device)
                    
                try:
                    attn_output = scaled_dot_product_attention(
                        query, key_cache, value_cache, is_causal=False
                    )
                except Exception as e:
                    raise
                attn = attn_output.squeeze(1).squeeze(1)
                del attn_output  # No longer needed after squeezing
                query = query.squeeze(1).squeeze(1)
        else:
            attn_output, _ = self.multihead_attn(
                query, key_cache, value_cache, is_causal=False
            )
            attn = attn_output.squeeze(1)
            del attn_output  # No longer needed after squeezing
            query = query.squeeze(1)
            
        return attn, query
    
    def intermediate_layers(self, i, emb, query, attn, hidden):
        intermediate_input = torch.cat((emb[i], query, attn, hidden[-1]), -1)
        del query, attn  
        intermediate = self.drop(
            self.tanh(self.int_norm(self.intermediate_h(intermediate_input)))
        )
        del intermediate_input  
        final_output = self.drop(self.tanh(self.f_norm(self.final_h(intermediate))))
        del intermediate  
        key_cache_i, value_cache_i, hidden_i = final_output.split(self.nhid, dim=-1)
        del final_output
        return key_cache_i, value_cache_i, hidden_i
    
    def get_query(self, emb, hidden):

        combined = torch.cat((emb, hidden[-1]), -1)
        query = self.drop(self.tanh(self.q_norm(self.q(combined))))
        del combined  # No longer needed after creating query
        query = query.unsqueeze(1)
        return query
    
    def forward(self, observation, initial_cache, nheads,replace = None):
        seq_len = observation.size(0)
        hidden, key_cache, value_cache = initial_cache
      
        # 1. Encode observations
        emb = self.drop(self.encoder(observation))
        del observation  # No longer needed after encoding
        
        for i in range(seq_len):
            # 2. Concatenate with previous hidden state
            
            
            
            query = self.get_query(emb[i], hidden)
            
            attn, query = self.attention_layer(query, key_cache, value_cache, nheads)
            
            if replace == 'hidden' :
                hidden_replaced = torch.zeros_like(hidden)
                key_cache_i, value_cache_i, hidden_i = self.intermediate_layers(i, emb, query, attn, hidden_replaced)
            elif replace == 'attention' :
                attention_replaced = torch.zeros_like(attn)
                key_cache_i, value_cache_i, hidden_i = self.intermediate_layers(i, emb, query, attention_replaced, hidden)
            elif replace == 'query' :
                query_replaced = torch.zeros_like(query)
                key_cache_i, value_cache_i, hidden_i = self.intermediate_layers(i, emb, query_replaced, attn, hidden)
            elif replace == 'emb' :
                emb_replaced = torch.zeros_like(emb[i])
                key_cache_i, value_cache_i, hidden_i = self.intermediate_layers(i, emb_replaced, query, attn, hidden)
            else :
                key_cache_i, value_cache_i, hidden_i = self.intermediate_layers(i, emb, query, attn, hidden)
                
            key_cache, value_cache, hidden = self.update_cache(key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i, nheads)

            del key_cache_i, value_cache_i, hidden_i  # No longer needed after concatenation

        cache = (hidden, key_cache, value_cache)
        decoded = self.decoder(hidden[1:])

        return decoded, cache



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


def eval(model, test_dataloader):
    condition_accuracies = defaultdict(int)
    condition_counts = defaultdict(int)
    correct_pred = 0
    sentence_details = []
    model.eval()
    # Forward pass with hidden state update word by word
    with torch.no_grad():
        for batch in test_dataloader:
            out = None
            written = batch["sentence"]
            sentence = batch["encoded_sentence"]
            correct = batch["encoded_correct"]
            wrong = batch["encoded_wrong"]
            condition = batch["condition"]
            batch_size = sentence.size(0)

            sent = sentence[:, :5].transpose(0, 1)
            cache = model.init_cache(sent,1)  # regarder si on peut mettre du priming
            # for i in range(sent.shape[1]):
            out, cache = model(sent, cache, 1, replace=None)
            log_probs = torch.nn.functional.log_softmax(
                out, dim=-1
            )  # s(out.squeeze(0))
            # déja sur correct et wrong log probs, pas les même résultats que sur extract_predictions.py
            correct_log_probs = log_probs[
                -1, torch.arange(batch_size), correct
            ]  # Shape: [512]
            wrong_log_probs = log_probs[-1, torch.arange(batch_size), wrong]
            correct_predictions = correct_log_probs >= wrong_log_probs

            for i in range(batch_size):
                cond = condition[i]
                pred = correct_predictions[i].item()  # Convert tensor to Python boolean
                condition_counts[cond] += 1
                condition_accuracies[cond] += pred

                sentence_details.append(
                    {
                        "sentence": written[i],
                        "condition": condition[i],
                        "correct_log_prob": correct_log_probs[i],
                        "wrong_log_prob": wrong_log_probs[i],
                        "model_prefers_correct": pred,
                    }
                )

    final_accuracies = {
        cond: condition_accuracies[cond] / condition_counts[cond]
        for cond in condition_accuracies
    }
    return final_accuracies


test_dataset = NounPPDataset(nounpp, dictionary)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
accuracies = []

checkpoint_files = [
    f
    for f in os.listdir(checkpoint_dir)
    if f.startswith("epoch_") and f.endswith(".pt")
]
accuracies_list = []


# Sort the checkpoint files based on the epoch number
def get_epoch_number(filename):
    match = re.search(r"epoch_(\d+)\.pt", filename)
    return int(match.group(1)) if match else 0


checkpoint_files.sort(key=get_epoch_number)

# Iterate through the sorted checkpoint files
for checkpoint_file in tqdm.tqdm(checkpoint_files, desc="Evaluating checkpoints"):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    model = CBR_RNN(
        ntoken=50001,
        ninp=args.emsize,
        nhid=args.nhid,
        device=device,
        nheads=args.nheads,
        dropout=0.2,
    )
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, map_location="cuda" if device == "cuda" else "cpu")
        model.load_state_dict(state_dict['model_state_dict'])
    model.to(device)

    acc = eval(model, test_dataloader)
    epoch_number = get_epoch_number(checkpoint_file)
    accuracies_list.append({"epoch": epoch_number, **acc})

df = pd.DataFrame(accuracies_list)
print(f"Saving results to: {output_path}")
df.to_csv(output_path, index=False)
print(df)

# to run on cpu : directly copy paste this in terminal (single head, hidden dim=embedding dim = 512)
# python /scratch2/mrenaudin/colorlessgreenRNNs/evaluation_notebooks/evaluate_cbr_on_nounpp.py --emsize 512 --nhid 512 --nheads 1 --checkpoint_dir '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/single_cbr_512' --output_name 'single_cbr_512'
# to run with 2 heads and dim=128
# python /scratch2/mrenaudin/colorlessgreenRNNs/evaluation_notebooks/evaluate_cbr_on_nounpp.py --emsize 128 --nhid 128 --nheads 2 --checkpoint_dir '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/multihead_cbr' --output_name '2_heads_cbr_128'
# to run with 8 heads
# python /scratch2/mrenaudin/colorlessgreenRNNs/evaluation_notebooks/evaluate_cbr_on_nounpp.py --emsize 128 --nhid 128 --nheads 8 --checkpoint_dir '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/8_heads_cbr' --output_name '8_heads_cbr_128'
#python /scratch2/mrenaudin/colorlessgreenRNNs/evaluation_notebooks/evaluate_cbr_on_nounpp.py --emsize 512 --nhid 512 --nheads 8 --checkpoint_dir '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/cbr8h512_shuffling' --output_name 'cbr8h512shuffling'

#python /scratch2/mrenaudin/colorlessgreenRNNs/evaluation_notebooks/evaluate_cbr_on_nounpp.py --emsize 128 --nhid 128 --nheads 1 --checkpoint_dir '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/cbr_1h_128_gumbel_softmax' --output_name '1head_128_gumbel_softmax'
# python /scratch2/mrenaudin/colorlessgreenRNNs/evaluation_notebooks/evaluate_cbr_on_nounpp.py --emsize 128 --nhid 128 --nheads 8 --checkpoint_dir '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/cbr8h128_shuffling' --output_name '8h_cbr_128_shuffling'