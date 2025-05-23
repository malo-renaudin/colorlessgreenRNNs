import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level from the script's directory (evaluation_notebooks)
parent_dir = os.path.dirname(script_dir)
# Join with 'src' to get the correct path to the src directory
src_dir = os.path.join(parent_dir, "src")
sys.path.append(os.path.abspath(src_dir))

from datasets import load_dataset
from language_models.model import RNNModel as lstm
import torch
from language_models.dictionary_corpus import Dictionary, Corpus, tokenize
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import string
import argparse
import re
import tqdm
import pandas as pd

parser = argparse.ArgumentParser(description="Evaluation of LSTM on Blimp")
parser.add_argument("--emsize", type=int, help="size of word embeddings")
parser.add_argument("--nhid", type=int, help="size of hidden state")
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
parser.add_argument(
    "--batch_size",
    type=int,
    default = 1024
)
args = parser.parse_args()
# general parameters and files for the eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"
dictionary = Dictionary(data_path)
checkpoint_dir = args.checkpoint_dir
output_dir = args.output_dir
output_name = args.output_name

blimp_tasks = ['adjunct_island', 
               'anaphor_gender_agreement', 
               'anaphor_number_agreement', 
               'animate_subject_passive', 
               'animate_subject_trans', 
               'causative', 
               'complex_NP_island', 
               'coordinate_structure_constraint_complex_left_branch', 
               'coordinate_structure_constraint_object_extraction', 
               'determiner_noun_agreement_1', 
               'determiner_noun_agreement_2', 
               'determiner_noun_agreement_irregular_1', 
               'determiner_noun_agreement_irregular_2', 
               'determiner_noun_agreement_with_adj_2', 
               'determiner_noun_agreement_with_adj_irregular_1', 
               'determiner_noun_agreement_with_adj_irregular_2', 
               'determiner_noun_agreement_with_adjective_1', 
               'distractor_agreement_relational_noun', 
               'distractor_agreement_relative_clause', 
               'drop_argument', 
               'ellipsis_n_bar_1', 
               'ellipsis_n_bar_2', 
               'existential_there_object_raising', 
               'existential_there_quantifiers_1', 
               'existential_there_quantifiers_2', 
               'existential_there_subject_raising', 
               'expletive_it_object_raising', 
               'inchoative', 
               'intransitive', 
               'irregular_past_participle_adjectives', 
               'irregular_past_participle_verbs', 
               'irregular_plural_subject_verb_agreement_1', 
               'irregular_plural_subject_verb_agreement_2', 
               'left_branch_island_echo_question', 
               'left_branch_island_simple_question', 
               'matrix_question_npi_licensor_present', 
               'npi_present_1', 
               'npi_present_2', 
               'only_npi_licensor_present', 
               'only_npi_scope', 
               'passive_1', 
               'passive_2', 
               'principle_A_c_command', 
               'principle_A_case_1', 
               'principle_A_case_2', 
               'principle_A_domain_1', 
               'principle_A_domain_2', 
               'principle_A_domain_3', 
               'principle_A_reconstruction', 
               'regular_plural_subject_verb_agreement_1', 
               'regular_plural_subject_verb_agreement_2', 
               'sentential_negation_npi_licensor_present', 
               'sentential_negation_npi_scope', 
               'sentential_subject_island', 
               'superlative_quantifiers_1', 
               'superlative_quantifiers_2', 
               'tough_vs_raising_1', 
               'tough_vs_raising_2', 
               'transitive', 
               'wh_island', 
               'wh_questions_object_gap', 
               'wh_questions_subject_gap', 
               'wh_questions_subject_gap_long_distance', 
               'wh_vs_that_no_gap', 
               'wh_vs_that_no_gap_long_distance', 
               'wh_vs_that_with_gap', 
               'wh_vs_that_with_gap_long_distance']

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_name)


class BLiMPDataset(Dataset):
    def __init__(self, blimp_subset, dictionary):
        self.dataset = load_dataset("nyu-mll/blimp", blimp_subset, split = 'train')
        self.dictionary = dictionary
        self.encoded_pairs = []

        for example in self.dataset:
            sentence_good = example['sentence_good']
            sentence_bad = example['sentence_bad']
            sentence_good = sentence_good.rstrip(string.punctuation)
            sentence_bad = sentence_bad.rstrip(string.punctuation)
            encoded_good = [self.dictionary.word2idx.get(word, self.dictionary.word2idx.get("<unk>")) for word in sentence_good.split()]
            #encoded_good = [self.dictionary.word2idx.get(word) for word in sentence_good.split()]
            encoded_bad = [self.dictionary.word2idx.get(word, self.dictionary.word2idx.get("<unk>")) for word in sentence_bad.split()]
            #encoded_bad = [self.dictionary.word2idx.get(word) for word in sentence_bad.split()]
            self.encoded_pairs.append({
                "sentence_good": sentence_good,
                "sentence_bad": sentence_bad,
                "encoded_good": torch.tensor(encoded_good, dtype=torch.long),
                "encoded_bad": torch.tensor(encoded_bad, dtype=torch.long),
            })

    def __len__(self):
        return len(self.encoded_pairs)

    def __getitem__(self, idx):
        return self.encoded_pairs[idx]
    
def collate_fn(batch):
    encoded_good_sequences = [item['encoded_good'] for item in batch]
    encoded_bad_sequences = [item['encoded_bad'] for item in batch]
    sentence_good = [item['sentence_good'] for item in batch]
    sentence_bad = [item['sentence_bad'] for item in batch]
    return {
        'sentence_bad':sentence_bad,
        'sentence_good': sentence_good,
        'encoded_good': pad_sequence(encoded_good_sequences, batch_first=True),
        'encoded_bad': pad_sequence(encoded_bad_sequences, batch_first=True)
    }

def compute_seq_nll(model,data, hidden, batch_size):
    batch_size, seq_len = data.shape
    #mask
    mask = (data!=0).float()
    #forward pass
    
    pad = data.swapaxes(0,1)
    
    output, hidden = model(pad, hidden)
    #target
    targets = data[:, 1:]#.swapaxes(0,1)
    #log probs
    log_probs = F.log_softmax(output, dim=-1)
    log_probs = log_probs[:-1]
    log_probs = log_probs.swapaxes(0,1)
    #nll loss
    #WITH NLL LOSS
    nll_loss = F.nll_loss(
            log_probs.reshape(-1, log_probs.size(-1)),
            targets.reshape(-1),
            reduction='none'
        )#.reshape(batch_size, max_len - 1)
    nll_loss=nll_loss.reshape(batch_size, seq_len - 1)
    #mask loss
    masked_nll_loss = nll_loss * mask[:, 1:]
    # Sum the negative log-likelihood over the sequence for each example
    sequence_nll = masked_nll_loss.sum(dim=1)
    return -sequence_nll


def eval(model, test_dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    #Forward pass with hidden state update word by word
    with torch.no_grad():
        for batch in test_dataloader:

            sentence_good = batch['sentence_good']
            sentence_bad = batch['sentence_bad']

            good = batch['encoded_good']
            bad = batch['encoded_bad']
        
            batch_size = good.size(0)

            hidden_good = model.init_hidden(batch_size)
            hidden_bad = model.init_hidden(batch_size)        
            seq_nll_good = compute_seq_nll(model,good, hidden_good, batch_size)
            seq_nll_bad = compute_seq_nll(model,bad, hidden_bad, batch_size)
            predictions = (seq_nll_good > seq_nll_bad).cpu().numpy()
            correct_predictions += np.sum(predictions)
            total_predictions += batch_size
      
            
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy on {test_dataloader.dataset.dataset.config_name}: {accuracy * 100:.2f}%")
    return accuracy
    
def main():
    # Find checkpoint files
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("epoch_") and f.endswith(".pt")
    ]
    print(f"Found {len(checkpoint_files)} checkpoints to evaluate")
    
    # No need to sort as the list is already in the desired order
    results = []
    
    # Iterate through the sorted checkpoint files
    for checkpoint_file in tqdm.tqdm(checkpoint_files, desc="Evaluating checkpoints"):
        print(f"\nEvaluating checkpoint: {checkpoint_file}")
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        model = lstm("LSTM", len(dictionary), args.emsize, args.nhid, 2, 0.2, False).to(device)
        
        with open(checkpoint_path, "rb") as f:
            state_dict = torch.load(f, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
        
        # Extract epoch and batch information from the filename
        if 'batch' in checkpoint_file:
            # Format: epoch_X_batch_Y.pt
            parts = checkpoint_file.replace('.pt', '').split('_')
            epoch_number = int(parts[1])
            batch_number = int(parts[3])
        else:
            # Format: epoch_X.pt
            epoch_number = int(checkpoint_file.replace('epoch_', '').replace('.pt', ''))
            batch_number = -1  # Use -1 to indicate it's a full epoch checkpoint
        
        checkpoint_results = {
            "epoch": epoch_number,
            "batch": batch_number,
            "checkpoint": checkpoint_file
        }
        
        for task in blimp_tasks:
            try:
                blimp_dataset = BLiMPDataset(task, dictionary)
                test_dataloader = DataLoader(
                    blimp_dataset, 
                    batch_size=args.batch_size, 
                    collate_fn=collate_fn
                )
                accuracy = eval(model, test_dataloader)
                checkpoint_results[task] = accuracy
            except Exception as e:
                print(f"Error evaluating task {task}: {e}")
                checkpoint_results[task] = float('nan')
        
        results.append(checkpoint_results)
        
        # Save results after each checkpoint to avoid losing data if the script crashes
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Results updated in {output_path}")

if __name__ == "__main__":
    main()

# to run on cpu : directly copy paste this in terminal
# python /scratch2/mrenaudin/colorlessgreenRNNs/evaluation_notebooks/evaluate_lstm_on_blimp.py --emsize 650 --nhid 650 --checkpoint_dir '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/lstm_adam_full_check' --output_name 'lstm_adam_full_check_blimp'
