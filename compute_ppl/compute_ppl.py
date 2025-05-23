import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level from the script's directory (evaluation_notebooks)
parent_dir = os.path.dirname(script_dir)
# Join with 'src' to get the correct path to the src directory
src_dir = os.path.join(parent_dir, "src")
sys.path.append(os.path.abspath(src_dir))

from language_models.dictionary_corpus import Corpus, TextDataset, Vocabulary, word_tokenizer, collate_batch, tokenize
from language_models.utils import (
    repackage_hidden,
    get_batch,
    batchify,
    save_checkpoint,
    move_to_device,
    save_val_loss_data,
    load_model,
    get_memory_usage,
    log_memory_usage,
    clear_memory,
    TemperatureScheduler,
    pick_lt_st_indices
)
import torch
from language_models.model import RNNModel as lstm
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import pickle


corpus = Corpus('/scratch2/mrenaudin/colorlessgreenRNNs/english_data')
ntokens = len(corpus.dictionary)

device = torch.device("cuda")



# Prepare data with batchify
eval_batch_size = 10

# Use regular batchify for all data
train_data = batchify(corpus.train, 512, device)
val_data = batchify(corpus.valid, eval_batch_size, device)
test_data = batchify(corpus.test, eval_batch_size, device)


def evaluate(model, data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = move_to_device(model.init_hidden(eval_batch_size), device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, 35):
            data, targets = get_batch(data_source, i, 35)
            data, targets = data.to(device), targets.to(device)
            
            
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += (
                len(data) * nn.CrossEntropyLoss()(output_flat, targets).item()
            )
            del output, output_flat
            hidden = repackage_hidden(hidden)

    return total_loss / (len(data_source) - 1)

checkpoint_dir_str = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/lstm_adam_full_check"
checkpoint_files =  [f'epoch_1_batch_{i}.pt' for i in range(0, 501, 1)]+[f'epoch_1_batch_{i}.pt'for i in range(200, 9300, 100)]
checkpoint_dir = Path(checkpoint_dir_str)

evals=[]
for item_name in tqdm(checkpoint_files, desc="Evaluating Checkpoints"):
    item_path = checkpoint_dir / item_name
    model = lstm('LSTM', 50001, 650, 650, 2, 0.2, False)
    with open(item_path, 'rb') as f:
        state_dict = torch.load(f, map_location='cuda' if device =='cuda' else 'cpu')
        model.load_state_dict(state_dict['model_state_dict'])
    model.to(device)
    model.eval() 
    evaluation = evaluate(model, val_data)
    evals.append(evaluation)
    
with open('evals.pkl', 'wb') as f:
    pickle.dump(evals, f)