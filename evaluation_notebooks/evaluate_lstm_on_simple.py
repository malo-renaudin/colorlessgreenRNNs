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
from torch.utils.data import Dataset, DataLoader
import tqdm
from language_models.dictionary_corpus import Dictionary, Corpus, tokenize
from language_models.model import RNNModel as lstm
from language_models.utils import move_to_device
import random
import pandas as pd
from collections import defaultdict
import numpy as np
from collections import defaultdict
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluation of LSTM on NounPP")
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
args = parser.parse_args()


batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"  # current directory
dictionary = Dictionary(data_path)
simple = '/scratch2/mrenaudin/colorlessgreenRNNs/simple.txt'
checkpoint_dir = args.checkpoint_dir
output_dir = args.output_dir
output_name = args.output_name

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_name)


class SimplePairDataset(Dataset):
    """
    Dataset that reads the simple file format:
    Sentence \t condition \t label(correct/wrong) \t id
    Groups pairs by id: for each id, stores (correct_sentence, wrong_sentence, condition)
    """
    def __init__(self, filepath, dictionary, device):
        self.dictionary = dictionary
        self.pairs = []  # list of dicts with keys: 'correct_sent', 'wrong_sent', 'condition'

        # Temp storage for grouping by id
        data_by_id = defaultdict(dict)

        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 4:
                    continue  # skip malformed lines
                sentence_str, condition, label, pair_id = parts
                # Tokenize the sentence (split by spaces)
                sentence_tokens = sentence_str.split()

                # Encode sentence words to indices, use <unk> if not found
                encoded_sentence = [
                    self.dictionary.word2idx.get(w, self.dictionary.word2idx.get("<unk>"))
                    for w in sentence_tokens
                ]

                data_by_id[pair_id][label] = {
                    "sentence": sentence_tokens,
                    "encoded_sentence": torch.tensor(encoded_sentence, dtype=torch.long),
                }
                data_by_id[pair_id]["condition"] = condition

        # Now create list of pairs
        for pair_id, val in data_by_id.items():
            if "correct" in val and "wrong" in val:
                self.pairs.append({
                    "correct_sentence": val["correct"]["sentence"],
                    "encoded_correct_sentence": val["correct"]["encoded_sentence"],
                    "wrong_sentence": val["wrong"]["sentence"],
                    "encoded_wrong_sentence": val["wrong"]["encoded_sentence"],
                    "condition": val["condition"],
                })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            "correct_sentence": pair["correct_sentence"],
            "encoded_correct_sentence": pair["encoded_correct_sentence"],
            "wrong_sentence": pair["wrong_sentence"],
            "encoded_wrong_sentence": pair["encoded_wrong_sentence"],
            "condition": pair["condition"],
        }

def collate_fn_pairs(batch):
    # batch is list of dicts as returned by __getitem__
    return {
        "correct_sentence": [item["correct_sentence"] for item in batch],
        "encoded_correct_sentence": torch.nn.utils.rnn.pad_sequence(
            [item["encoded_correct_sentence"] for item in batch], batch_first=True
        ),
        "wrong_sentence": [item["wrong_sentence"] for item in batch],
        "encoded_wrong_sentence": torch.nn.utils.rnn.pad_sequence(
            [item["encoded_wrong_sentence"] for item in batch], batch_first=True
        ),
        "condition": [item["condition"] for item in batch],
    }

init_sentence = " ".join(["In service , the aircraft was operated by a crew of five and could accommodate either 30 paratroopers , 32 <unk> and 28 sitting casualties , or 50 fully equipped troops . <eos>",
                    "He even speculated that technical classes might some day be held \" for the better training of workmen in their several crafts and industries . <eos>",
                    "After the War of the Holy League in 1537 against the Ottoman Empire , a truce between Venice and the Ottomans was created in 1539 . <eos>",
                    "Moore says : \" Tony and I had a good <unk> and off-screen relationship , we are two very different people , but we did share a sense of humour \" . <eos>",
                    "<unk> is also the basis for online games sold through licensed lotteries . <eos>"])

def feed_input(model, hidden, w):
    inp = torch.autograd.Variable(torch.LongTensor([[dictionary.word2idx[w]]])).to(device)
    out, hidden = model(inp, hidden)
    return out, hidden
def feed_sentence(model, h, sentence):
    outs = []
    for w in sentence:
        out, h = feed_input(model, h, w)
        outs.append(torch.nn.functional.log_softmax(out[0]).unsqueeze(0))
    return outs, h

def get_last_word_log_prob(model, encoded_sentences, hidden):
    out = None
    for t in range(encoded_sentences.size(1) - 1):
        inp = encoded_sentences[:, t].unsqueeze(0).to(device)
        out, hidden = model(inp, hidden)
    log_probs = torch.nn.functional.log_softmax(out, dim=-1)
    # Target word is the last word in sentence
    targets = encoded_sentences[:, -1].to(device)
    # Gather log-prob of target word
    last_word_log_prob = log_probs[0, torch.arange(encoded_sentences.shape[0]), targets]
    return last_word_log_prob.cpu()

def eval_pairs(model, dataloader, init_sentence):
    condition_accuracies = defaultdict(int)
    condition_counts = defaultdict(int)
    sentence_details = []

    model.eval()
    hidden = model.init_hidden(1)
    _, init_h = feed_sentence(model, hidden, init_sentence.split(" "))

    with torch.no_grad():
        for batch in dataloader:
            batch_size = len(batch["condition"])

            # Expand init hidden states for batch
            hidden = (
                init_h[0].expand(-1, batch_size, -1).contiguous(),
                init_h[1].expand(-1, batch_size, -1).contiguous(),
            )

            # Evaluate each sentence in the batch: get log-prob of last wor
            correct_log_probs = get_last_word_log_prob(model, batch["encoded_correct_sentence"], hidden)
            wrong_log_probs = get_last_word_log_prob(model, batch["encoded_wrong_sentence"], hidden)

            preds = correct_log_probs >= wrong_log_probs

            for i in range(batch_size):
                cond = batch["condition"][i]
                pred = preds[i].item()
                condition_counts[cond] += 1
                condition_accuracies[cond] += pred
                sentence_details.append({
                    "correct_sentence": batch["correct_sentence"][i],
                    "wrong_sentence": batch["wrong_sentence"][i],
                    "condition": cond,
                    "correct_log_prob": correct_log_probs[i].item(),
                    "wrong_log_prob": wrong_log_probs[i].item(),
                    "model_prefers_correct": pred,
                })

    final_accuracies = {cond: condition_accuracies[cond] / condition_counts[cond]
                       for cond in condition_accuracies}
    return final_accuracies, sentence_details

test_dataset = SimplePairDataset(simple, dictionary, device)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn_pairs)

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
        
        acc = eval_pairs(model, test_dataloader, init_sentence)
        checkpoint_results['accuracy']=acc[0]
        results.append(checkpoint_results)
        
        # Save results after each checkpoint to avoid losing data if the script crashes
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Results updated in {output_path}")

if __name__ == "__main__":
    main()
    
#python /scratch2/mrenaudin/colorlessgreenRNNs/evaluation_notebooks/evaluate_lstm_on_simple.py --emsize 650 --nhid 650 --checkpoint_dir '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/lstm_adam_full_check' --output_name 'lstm_adam_full_check_simple'
