
import torch
import torch.nn as nn
import math
import logging
from torch.nn.functional import scaled_dot_product_attention
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class CBR_RNN_attn_tracking(nn.Module):
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
    
    @staticmethod
    def temperature_attention(query, key, value, temperature, gumbel_softmax=None, attn_mask=None,dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        if gumbel_softmax: 
            attn_weight = torch.gumbel_softmax(attn_weight, tau=temperature, hard=False, dim=-1)
        else : 
            attn_weight = attn_weight/temperature
            attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight, attn_weight @ value
    
    def attention_layer(self, query, key_cache, value_cache, nheads, temperature, gumbel_softmax):
        if nheads == 1:
                query = query.unsqueeze(1)
                
                # Ensure all tensors are on the same device
                if query.device != key_cache.device:
                    key_cache = key_cache.to(query.device)
                if query.device != value_cache.device:
                    value_cache = value_cache.to(query.device)
                if temperature is None:  
                    try:
                        attn_output = scaled_dot_product_attention(
                            query, key_cache, value_cache, is_causal=False
                        )
                    except Exception as e:
                        logging.error(f"Error in attention computation: {str(e)}")
                        raise
                    attn = attn_output.squeeze(1).squeeze(1)
                    del attn_output  # No longer needed after squeezing
                    query = query.squeeze(1).squeeze(1)
                else:
                    try:
                        attn_weight, attn_output = self.temperature_attention(
                            query, key_cache, value_cache, temperature, is_causal=False
                        )
                    except Exception as e:
                        logging.error(f"Error in attention computation: {str(e)}")
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
            
        return attn_weight, attn, query
    
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
    
    def forward(self, observation, initial_cache, nheads, temperature, gumbel_softmax):
        seq_len = observation.size(0)
        hidden, key_cache, value_cache = initial_cache

        # 1. Encode observations
        emb = self.drop(self.encoder(observation))
        del observation  # No longer needed after encoding
        attn_weights = []
        for i in range(seq_len):
            # 2. Concatenate with previous hidden state

            
            query = self.get_query(emb[i], hidden)
            
            attn_weight, attn, query = self.attention_layer(query, key_cache, value_cache, nheads, temperature, gumbel_softmax)
            attn_weights.append(attn_weight)
            key_cache_i, value_cache_i, hidden_i = self.intermediate_layers(i, emb, query, attn, hidden)
            
            key_cache, value_cache, hidden = self.update_cache(key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i, nheads)
            
            del key_cache_i, value_cache_i, hidden_i  # No longer needed after concatenation

        decoded = self.decoder(hidden[1:])

        return decoded, hidden, attn_weights

class WMTestDataset(Dataset):
    def __init__(self, sentences_path, markers_path, dictionary):
        """
        sentences_path: path to the sentence file (one sentence per line)
        markers_path: path to marker file (CSV/TSV with columns: markers, stimid, list_len, prompt_len)
                      markers column contains array-like strings (e.g. '[0,0,1,1,...]')
        dictionary: object with attribute 'word2idx' (dict mapping words to indices), and
                    should contain an "<unk>" key for unknown words.
        """
        self.dictionary = dictionary
        
        # Load sentences
        with open(sentences_path, "r", encoding="utf-8") as f:
            self.sentences = [line.strip().split() for line in f if line.strip()]
        
        # Load markers metadata file with pandas
        self.markers_df = pd.read_csv(markers_path, sep="\t", converters={
            'markers': lambda x: list(map(int, x.strip('[]').split(',')))
        })
        
        # Sanity check: same number of sentences and markers rows
        assert len(self.sentences) == len(self.markers_df), \
            f"Mismatch: {len(self.sentences)} sentences vs {len(self.markers_df)} marker entries"
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        row = self.markers_df.iloc[idx]
        
        markers = row['markers']
        stimid = row['stimid']
        list_len = row['list_len']
        prompt_len = row['prompt_len']
        
        encoded_sentence = [
            self.dictionary.word2idx.get(word, self.dictionary.word2idx.get("<unk>"))
            for word in sentence
        ]
        
        list1 = [sentence[i] for i, tag in enumerate(markers) if tag == 1]
        list2 = [sentence[i] for i, tag in enumerate(markers) if tag == 3]
        
        list1_encoded = [
        encoded_sentence[i] for i, tag in enumerate(markers) if tag == 1
        
        ]
        list2_encoded = [
            encoded_sentence[i] for i, tag in enumerate(markers) if tag == 3
        ]
        
        condition = (list_len, prompt_len)
        
        sample = {
            "sentence": sentence,              # list of tokens (words)
            "encoded_sentence": encoded_sentence, 
            'list1_encoded':list1_encoded,
            "list1":list1,
            "list2_encoded": list2_encoded,
            "list2":list2,
            "stimid": stimid,  
            "condition" : condition,
            "marker":markers
            # "list_len": list_len,              # int
            # "prompt_len": prompt_len           # int
        }
        return sample

def collate_fn(batch):
    """Custom collate function to properly handle sentences as lists of strings."""
    sentences = [item["sentence"] for item in batch] 
    list1 = [item["list1"] for item in batch] 
    list2 = [item["list2"] for item in batch] 
    encoded_sentences = torch.stack([torch.tensor(item["encoded_sentence"]) for item in batch])  
    list1_encoded = torch.stack([torch.tensor(item["list1_encoded"]) for item in batch])
    list2_encoded = torch.stack([torch.tensor(item["list2_encoded"]) for item in batch])
    marker = torch.stack([torch.tensor(item['marker']) for item in batch])
    conditions = [item["condition"] for item in batch]

    return {
        "sentence": sentences,
        "encoded_sentence": encoded_sentences,
        "list1": list1,
        "list2": list2,
        "list1_encoded":list1_encoded,
        "list2_encoded":list2_encoded,
        "condition": conditions,
        "marker": marker
    }

def eval(model, dataloader, nheads, temperature, gs):
    all_repeat_surprisals = {}
    model.eval()
    # Forward pass with hidden state update word by word
    with torch.no_grad():
        for batch in dataloader:
            out = None
            list1 = batch["list1"]
            encoded_sentence = batch["encoded_sentence"]
            condition = batch["condition"]
            marker = batch["marker"]
            batch_size, seq_len = encoded_sentence.shape
            input_seq = encoded_sentence[:, :-1].transpose(0, 1)  # (seq_len-1, batch_size)
            target_seq = encoded_sentence[:, 1:].transpose(0, 1)
            cache = model.init_cache(input_seq, 1)
            output, hidden, attn_weights = model(input_seq, cache, nheads, temperature, gs)
            log_probs = F.log_softmax(output, dim=-1)  # shape (seq_len-1, batch_size, vocab_size)
            
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
            # print(word_indices)
            # for i in word_indices:
            #     words_at_indices = list1[0][i]
            #     print(words_at_indices)
            surprisal1_repeats = surprisal_list1[:, word_indices]
            surprisal2_repeats = surprisal_list2[:, word_indices]
            
            # Compute repeat surprisal ratio as percentage
            repeat_surprisal = (surprisal2_repeats / surprisal1_repeats) * 100
            all_repeat_surprisals[f'list len : {condition[0][0]}, prompt len : {condition[0][1]}']=repeat_surprisal
            break
    return all_repeat_surprisals, attn_weights, encoded_sentence
        
        
def plot_attention(att_matrix, sentence):

    # Example data (replace with your actual data)
    att_matrix_np = att_matrix.numpy()  # Shape: [seq_len, seq_len]
    tokens = sentence  # Your list of tokens, length seq_len + 1 (make sure sizes match)

    # If att_matrix_np shape is smaller than tokens length,
    # slice tokens to match matrix size, or vice versa:
    tokens = tokens[:att_matrix_np.shape[0]]

    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the heatmap
    cax = ax.matshow(att_matrix_np, cmap='brg', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))

    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    # Add colorbar
    fig.colorbar(cax, label='Attention Weight')

    ax.set_xlabel('Key (attended word)')
    ax.set_ylabel('Query (current word)')
    ax.set_title('Attention Heatmap')

    plt.tight_layout()
    plt.show()