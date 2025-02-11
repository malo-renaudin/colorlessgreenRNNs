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
from dictionary_corpus import Corpus
import model
from lm_argparser import lm_parser
from utils import repackage_hidden, get_batch, batchify, save_checkpoint, move_to_device, save_val_loss_data
import torch.profiler

parser = argparse.ArgumentParser(parents=[lm_parser],
                                 description="Basic training and evaluation for RNN LM")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                                                  logging.FileHandler(args.log)])
logging.info(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
#NEW : added device      
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

logging.info("Batchying..")
eval_batch_size = 10
#NEW : changed args.cuda to device
train_data = batchify(corpus.train, args.batch_size, device)
val_data = batchify(corpus.valid, eval_batch_size, device)
test_data = batchify(corpus.test, eval_batch_size, device)



criterion = nn.CrossEntropyLoss()

###############################################################################
# Build the model
###############################################################################

logging.info("Building the model")

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
#NEW : changed model.cuda() to model.to(device)
if args.cuda:
    #model.cuda()
    model = model.to(device)



###############################################################################
# Training code
###############################################################################


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    #NEW : move hidden to device
    hidden = move_to_device(model.init_hidden(eval_batch_size), device)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
            #NEW : move data and targets to device
            data, targets = data.to(device), targets.to(device)
            #> output has size seq_length x batch_size x vocab_size
            output, hidden = model(data, hidden)
            #> output_flat has size num_targets x vocab_size (batches are stacked together)
            #> ! important, otherwise softmax computation (e.g. with F.softmax()) is incorrect
            output_flat = output.view(-1, ntokens)
            #output_candidates_info(output_flat.data, targets.data)
            total_loss += len(data) * nn.CrossEntropyLoss()(output_flat, targets).item()
            hidden = repackage_hidden(hidden)

    return total_loss / (len(data_source) - 1)
#NEW : create folder for checkpointing
main_folder = '/scratch2/mrenaudin/colorlessgreenRNNs/val_loss'
subfolder = os.path.join(main_folder, args.name)
os.makedirs(subfolder, exist_ok=True)
val_loss_data = []

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    #NEW : move hidden to devide
    hidden = move_to_device(model.init_hidden(args.batch_size), device)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)
        #NEW : move data and target to device
        data, targets = data.to(device), targets.to(device)

        # truncated BPP
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()
        #NEW : added checkpointing
        #checkpointing every batch for the 5 first epochs
        if epoch <= 3 and batch % 100 == 0:
            save_checkpoint(model, args.name, epoch, batch)
            val_loss = evaluate(val_data)
            filename = f'epoch{epoch}_batch{batch}'
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | val_loss{:5.2f}'.format(epoch, batch, len(train_data) // args.bptt, val_loss))
            val_loss_data.append({'epoch': epoch, 'batch': batch, 'val_loss': val_loss})
            save_val_loss_data(val_loss_data, subfolder, filename)
            model.train()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train()

        val_loss = evaluate(val_data)
        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        logging.info('-' * 89)
        
        #NEW : added checkpointing
        #checkpointing every epochs after the 5th epoch
        if epoch > 3:
            save_checkpoint(model, args.name, epoch)
            # val_loss = evaluate(val_data)
            # logging.info('| epoch {:3d} | val_loss{:5.2f}'.format(epoch, val_loss))
            val_loss_data.append({'epoch': epoch, 'batch': 'end_of_epoch', 'val_loss': val_loss})
            filename = f'epoch{epoch}'
            save_val_loss_data(val_loss_data, subfolder, filename)
            model.train()
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting from training early')
    
# val_loss_df = pd.DataFrame(val_loss_data)
# val_loss_df.to_csv('val_loss.csv', index=False)

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
logging.info('=' * 89)
logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
logging.info('=' * 89)
