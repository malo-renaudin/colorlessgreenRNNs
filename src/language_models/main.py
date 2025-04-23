# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import math
import time
import pandas as pd
import torch
import torch.nn as nn
import os
import gc
import psutil
from dictionary_corpus import Corpus
import model
from lm_argparser import lm_parser
from utils import (
    repackage_hidden,
    get_batch,
    batchify,
    save_checkpoint,
    move_to_device,
    save_val_loss_data,
    load_model,
)
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity


parser = argparse.ArgumentParser(
    parents=[lm_parser], description="Basic training and evaluation for RNN LM"
)

args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler(args.log)],
)
logging.info(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
# NEW : added device
device = torch.device("cuda" if args.cuda else "cpu")
print(f"Using device: {device}")

###############################################################################
# Load data
###############################################################################

logging.info("Loading data")
start = time.time()
corpus = Corpus(args.data)
logging.info("( %.2f )" % (time.time() - start))
ntokens = len(corpus.dictionary)
logging.info("Vocab size %d", ntokens)

# Batchify the data
logging.info("Batchying..")
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size, device)
val_data = batchify(corpus.valid, eval_batch_size, device)
test_data = batchify(corpus.test, eval_batch_size, device)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Build the model
###############################################################################

logging.info("Building the model")
print(args.model)
model, optimizer_state_dict = load_model(
    args.classmodel,
    args.model,
    ntokens,
    args.emsize,
    args.nhid,
    args.nheads,
    args.dropout,
    device,
    args.nlayers,
    args.tied,
    args.checkpoint_path,
)
#optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optim.SGD(model.parameters(), lr=args.lr)
if optimizer_state_dict is not None:
    optimizer.load_state_dict(optimizer_state_dict)
    logging.info("Loaded optimizer state from checkpoint")

###############################################################################
# Training code
###############################################################################

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

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    if args.classmodel != "CBR_RNN":
        hidden = move_to_device(model.init_hidden(eval_batch_size), device)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
            data, targets = data.to(device), targets.to(device)
            
            if args.classmodel == "CBR_RNN":
                cache = model.init_cache(data, args.nheads)
                output, hidden = model(data, cache, args.nheads)
                output_flat = output.reshape(-1, output.size(-1))
                targets_flat = targets.reshape(-1)
                total_loss += (
                    len(data) * nn.CrossEntropyLoss()(output_flat, targets_flat).item()
                )
                del output, output_flat, targets_flat, cache
            else:
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += (
                    len(data) * nn.CrossEntropyLoss()(output_flat, targets).item()
                )
                del output, output_flat
                hidden = repackage_hidden(hidden)
            #clear_memory()

    return total_loss / (len(data_source) - 1)


# NEW : create folder for checkpointing
main_folder = "/scratch2/mrenaudin/colorlessgreenRNNs/val_loss"
subfolder = os.path.join(main_folder, args.name)
os.makedirs(subfolder, exist_ok=True)
val_loss_data = []

def train():
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # Log initial memory usage
    #log_memory_usage("Initial ")
    
    # Initialize cache once per epoch for CBR_RNN
    if args.classmodel == "CBR_RNN":
        # Get first batch to determine dimensions
        first_batch, _ = get_batch(train_data, 0, args.bptt)
        first_batch = first_batch.to(device)
        cache = model.init_cache(first_batch, args.nheads)
        del first_batch  # Clean up the temporary batch
        #clear_memory()
    else:
        hidden = move_to_device(model.init_hidden(args.batch_size), device)
    
    if epoch == 1:
        save_checkpoint(model, optimizer, args.name, epoch, 0)
        logging.info(f"Checkpoint saved before the first batch: {epoch}, batch {0}")

    # Initialize PyTorch profiler
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA if torch.cuda.is_available() else ProfilerActivity.CPU
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_logs/{args.name}'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            with record_function("batch_processing"):
                data, targets = get_batch(train_data, i, args.bptt)
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()

                if args.classmodel == "CBR_RNN":
                    with record_function("forward_pass"):
                        output, hidden = model(data, cache, args.nheads)
                        output_flat = output.reshape(-1, output.size(-1))
                        targets_flat = targets.reshape(-1)
                        loss = criterion(output_flat, targets_flat)
                        
                        del output, output_flat, targets_flat
                else:
                    with record_function("forward_pass"):
                        hidden = repackage_hidden(hidden)
                        output, hidden = model(data, hidden)
                        loss = criterion(output.view(-1, ntokens), targets)
                        del output

                if torch.isnan(loss):
                    raise ValueError("NaN loss encountered")
                
                try:
                    with record_function("backward_pass"):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        optimizer.step()
                    
                except RuntimeError as e:
                    logging.error(f"Error during backward pass: {str(e)}")
                    logging.error(f"Loss value: {loss.item()}")
                    logging.error(f"Gradients before backward:")
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            logging.error(f"{name} - grad shape: {param.grad.shape}")
                    raise e

        total_loss += loss.item()
        #del loss
        #clear_memory()

        # Checkpoint and validation
        # if args.checkpoint_path and batch % args.batch_check == 0:
        #     save_checkpoint(model, optimizer, args.name, epoch, batch)
        #     val_loss = evaluate(val_data)
        #     filename = f"epoch{epoch}_batch{batch}"
        #     logging.info(
        #         "| epoch {:3d} | {:5d}/{:5d} batches | val_loss{:5.2f}".format(
        #             epoch, batch, len(train_data) // args.bptt, val_loss
        #         )
        #     )
        #     val_loss_data.append(
        #         {"epoch": epoch, "batch": batch, "val_loss": val_loss}
        #     )
        #     save_val_loss_data(val_loss_data, subfolder, filename)
        #     model.train()  # Set back to training mode after evaluation
        #     clear_memory()

            
                # Logging
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging.info(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // args.bptt,
                    args.lr,  # Use initial learning rate
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()
        
        prof.step()  # Step the profiler

# Loop over epochs.
try:
    if args.epoch_checkpointed:
        k = int(args.epoch_checkpointed)
    else:
        k = 1
        
    for epoch in range(k, args.epochs + 1):
        epoch_start_time = time.time()
        train()

        val_loss = evaluate(val_data)
        logging.info("-" * 89)
        logging.info(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
            "valid ppl {:8.2f}".format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
            )
        )
        logging.info("-" * 89)

        # Save checkpoint at end of epoch
        save_checkpoint(model, optimizer, args.name, epoch)
        val_loss_data.append(
            {"epoch": epoch, "batch": "end_of_epoch", "val_loss": val_loss}
        )
        filename = f"epoch{epoch}"
        save_val_loss_data(val_loss_data, subfolder, filename)
        model.train()  # Set back to training mode after evaluation

except KeyboardInterrupt:
    logging.info("-" * 89)
    logging.info("Exiting from training early")

# val_loss_df = pd.DataFrame(val_loss_data)
# val_loss_df.to_csv('val_loss.csv', index=False)

# Load the best saved model.
with open(args.save, "rb") as f:
    model = torch.load(f)

# Run on test data
test_loss = evaluate(test_data)
logging.info("=" * 89)
logging.info(
    "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
        test_loss, math.exp(test_loss)
    )
)
logging.info("=" * 89)

