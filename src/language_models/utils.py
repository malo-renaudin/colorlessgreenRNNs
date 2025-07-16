# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import os
import logging
import pandas as pd
import psutil
import gc
import numpy as np 
import random
import torch.nn as nn
import math

def repackage_hidden(h):
    """Detaches hidden states from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, seq_length):
    """Gets a single batch from source data at position i"""
    seq_len = min(seq_length, len(source) - 1 - i)
    data = source[i : i + seq_len]
    # predict the sequences shifted by one word
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    # if device = 'cuda':
    #     #data = data.cuda()
    data = data.to(device)
    return data


def shuffled_batchify(data, bsz, device):
    """Similar to batchify but shuffles data first"""
    # Get the original data size
    data_size = data.size(0)
    
    # Create shuffled indices
    indices = torch.randperm(data_size)
    
    # Shuffle the data using indices
    shuffled_data = data[indices]
    
    # Now batchify as usual
    nbatch = shuffled_data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit
    shuffled_data = shuffled_data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches
    shuffled_data = shuffled_data.view(bsz, -1).t().contiguous()
    
    # Move to device
    shuffled_data = shuffled_data.to(device)
    
    return shuffled_data


# inclure batch index en argument ici comme ça on désengorge script principal
def save_checkpoint(model, optimizer, experiment_name, epoch, temperature, checkpoint_dir, batch=None):
    """Save model checkpoint."""
    #checkpoint_dir = "checkpoints"

    # Create a subfolder for the experiment within the checkpoints directory
    experiment_dir = os.path.join(checkpoint_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    # if batch is None or batch % 10 == 0:

    if batch is None:
        filename = f"{experiment_dir}/epoch_{epoch}.pt"
    else:
        filename = f"{experiment_dir}/epoch_{epoch}_batch_{batch}.pt"
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "temperature": temperature
    }
    torch.save(checkpoint, filename)
    logging.info(f"Checkpoint saved: {filename}")


def move_to_device(hidden, device):
    """Move each tensor in the hidden state tuple to the specified device."""
    if isinstance(hidden, torch.Tensor):
        return hidden.to(device)
    else:
        return tuple(move_to_device(h, device) for h in hidden)


def save_val_loss_data(val_loss_data, folder, filename):
    val_loss_df = pd.DataFrame(val_loss_data)
    val_loss_df.to_csv(os.path.join(folder, filename), index=False)


def load_model(
    classmodel,
    model,
    ntokens,
    emsize,
    nhid,
    nheads,
    dropout,
    device,
    nlayers,
    tied,
    checkpoint_path,
    memory_size,
    memory_dim,
    bptt
):
    import model as m

    if classmodel == "RNNModel":
        model = m.RNNModel(model, ntokens, emsize, nhid, nlayers, dropout, tied)

    elif classmodel == "CBR_RNN":
        model = m.CBR_RNN(ntokens, emsize, nhid, nheads, bptt, dropout, device)
        
    elif classmodel == 'Stack_LSTM':
        model = m.Stack_LSTM(ntokens, emsize, nhid, device, nlayers, memory_size, memory_dim)
        

    optimizer_state_dict = None
    if checkpoint_path:
        with open(checkpoint_path, "rb") as f:
            state_dict = torch.load(
                f, map_location="cuda" if device == "cuda" else "cpu"
            )
            model.load_state_dict(state_dict["model_state_dict"])
            optimizer_state_dict = state_dict["optimizer_state_dict"]
            temperature = state_dict['temperature']

    model = model.to(device)
    return model, optimizer_state_dict#, temperature

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**2
        gpu_cache = torch.cuda.memory_reserved() / 1024**2
        return {
            'cpu_mem': process.memory_info().rss / 1024**2,
            'gpu_mem': gpu_mem,
            'gpu_cache': gpu_cache
        }
    return {'cpu_mem': process.memory_info().rss / 1024**2}

def log_memory_usage(prefix=""):
    """Log current memory usage"""
    mem = get_memory_usage()
    if torch.cuda.is_available():
        logging.info(f"{prefix}Memory Usage - CPU: {mem['cpu_mem']:.2f}MB, GPU: {mem['gpu_mem']:.2f}MB, GPU Cache: {mem['gpu_cache']:.2f}MB")
    else:
        logging.info(f"{prefix}Memory Usage - CPU: {mem['cpu_mem']:.2f}MB")

def clear_memory():
    """Clear both Python and CUDA memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class TemperatureScheduler:
    def __init__(self, start_temp=1.0, min_temp=0.1, total_steps = 40):
        self.temperature = start_temp
        self.min_temp = min_temp
        self.step_count = 0
        self.decay_rate = (min_temp / start_temp) ** (1 / total_steps)
        self.temperature = start_temp

    def step(self):
        self.temperature = max(self.min_temp, self.temperature * self.decay_rate)
        self.step_count += 1
        return self.temperature

    def get_temperature(self):
        return self.temperature
    
def pick_lt_st_indices(num_layers, hidden_size_per_layer, n_lt, n_st, seed=None):
    total_neurons = num_layers * hidden_size_per_layer
    if n_lt + n_st > total_neurons:
        raise ValueError(f"Cannot pick {n_lt} long-term and {n_st} short-term neurons from {total_neurons} total neurons.")

    if seed is not None:
        random.seed(seed)

    # Generate all (layer_idx, neuron_idx) pairs
    all_indices = [
        (layer, neuron)
        for layer in range(num_layers)
        for neuron in range(hidden_size_per_layer)
    ]

    # Sample
    lt_indices = random.sample(all_indices, n_lt)
    remaining_indices = list(set(all_indices) - set(lt_indices))
    st_indices = random.sample(remaining_indices, n_st)

    return lt_indices, st_indices

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as used in Transformer models"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0), :]