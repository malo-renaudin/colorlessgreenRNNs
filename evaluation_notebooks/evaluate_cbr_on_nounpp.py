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
    def __init__(self, ntoken, ninp, nhid, device, nheads, dropout=0.5):
        super().__init__()
        # same layers as Timkey
        self.device = device
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.score_attn = nn.Softmax(dim=-1)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.q = nn.Linear(ninp + nhid, nhid)
        self.intermediate_h = nn.Linear(nhid * 4, nhid * 4)
        self.decoder = nn.Linear(nhid, ntoken + 1)
        self.q_norm = torch.nn.LayerNorm(nhid)
        self.int_norm = torch.nn.LayerNorm(nhid * 4)
        self.f_norm = torch.nn.LayerNorm(nhid * 3)
        self.nhid = nhid
        self.attn_div_factor = np.sqrt(nhid)
        self.final_h = nn.Linear(nhid * 4, nhid * 3)
        self.nheads = nheads

        # for multihead attention
        if self.nheads > 1:
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=nhid, num_heads=nheads, batch_first=True
            )

    # same weight initialization as Timkey
    def init_weights(self, freeze_embedding, aux_objective):
        """Initialize encoder and decoder weights"""
        initrange = 0.1
        if not freeze_embedding:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if aux_objective:
            self.aux_decoder.bias.data.fill_(0)
            self.aux_decoder.weight.data.uniform_(-initrange, initrange)

    # def init_hidden(self, bsz):
    #     """Initialize a fresh hidden state"""
    #     weight = next(self.parameters()).data

    #     return torch.tensor(weight.new(bsz, self.nhid).zero_())

    def init_cache(self, observation):
        if len(observation.size()) > 1:
            bsz = observation.size(dim=-1)
        else:
            bsz = 1
        seq_len = observation.size(dim=0)

        return (
            torch.zeros(1, bsz, self.nhid).to(self.device),
            torch.zeros(1, bsz, self.nhid).to(self.device),
            torch.zeros(1, bsz, self.nhid).to(self.device),
        )

    def forward(self, observation, initial_cache, nheads):
        # Get dimensions
        seq_len = observation.size(0)  # if len(observation.size()) > 1 else 1

        observation = observation.to(self.device)
        # Unpack initial cache
        hidden, key_cache, value_cache = (
            initial_cache  # hidden is initialized as the query and updated at each time step
        )

        # 1. Encode observations
        emb = self.drop(self.encoder(observation))
        # Process sequence : is there another more efficient way to compute causal attention than looping ?
        for i in range(
            seq_len
        ):  # need to keep sequential processing as the core structure is recurrent (each new word needs the hidden state obtained after prediction of the last word)
            # 2. Concatenate with previous hidden state
            query = self.drop(
                self.tanh(self.q_norm(self.q(torch.cat((emb[i], hidden[i]), -1))))
            )  # b * d
            query = query.unsqueeze(1)
            if nheads == 1:
                attn_output = scaled_dot_product_attention(
                    query,
                    key_cache.transpose(0, 1),
                    value_cache.transpose(
                        0, 1
                    ),  # batch dimension needs to be the first one, hence the transpose (and the unsuqeeze on the query), second dim is seq len
                    is_causal=True,
                )
                attn = attn_output.squeeze(1)

                intermediate = self.drop(
                    self.tanh(
                        self.int_norm(
                            self.intermediate_h(
                                torch.cat(
                                    (emb[i], query.squeeze(1), attn, hidden[i]), -1
                                )
                            )
                        )
                    )
                )
                key_cache_i, value_cache_i, hidden_i = self.drop(
                    self.tanh(self.f_norm(self.final_h(intermediate)))
                ).split(self.nhid, dim=-1)

                hidden = torch.cat((hidden, hidden_i.unsqueeze(0)), dim=0)
                key_cache = torch.cat((key_cache, key_cache_i.unsqueeze(0)), dim=0)

                value_cache = torch.cat(
                    (value_cache, value_cache_i.unsqueeze(0)), dim=0
                )
            else:
                attn_output = self.multihead_attn(
                    query, key_cache.transpose(0, 1), value_cache.transpose(0, 1)
                )
                # outputs attn output and attn output weights
                attn = attn_output[0]  # .squeeze(1)
                intermediate = self.drop(
                    self.tanh(
                        self.int_norm(
                            self.intermediate_h(
                                torch.cat(
                                    (
                                        emb[i].unsqueeze(0),
                                        query.transpose(0, 1),
                                        attn.transpose(0, 1),
                                        hidden[i].unsqueeze(0),
                                    ),
                                    -1,
                                )
                            )
                        )
                    )
                )

                intermediate_2 = self.drop(
                    self.tanh(self.f_norm(self.final_h(intermediate)))
                )
                key_cache_i, value_cache_i, hidden_i = intermediate_2.split(
                    self.nhid, dim=-1
                )
                hidden = torch.cat((hidden, hidden_i), dim=0)
                key_cache = torch.cat((key_cache, key_cache_i), dim=0)
                value_cache = torch.cat((value_cache, value_cache_i), dim=0)

        decoded = self.decoder(hidden[1:])
        cache = [hidden, key_cache, value_cache]

        return decoded, hidden, cache


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
            cache = model.init_cache(sent)  # regarder si on peut mettre du priming
            # for i in range(sent.shape[1]):
            out, hidden, cache = model(sent, cache, 1)
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
        model.load_state_dict(state_dict)
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
