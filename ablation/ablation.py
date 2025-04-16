import sys
import os
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
import argparse
import re

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
    help="base name of the results dataframe file (will be appended with ablation info)",
)
args = parser.parse_args()

# general parameters and files for the eval
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"
dictionary = Dictionary(data_path)
nounpp = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt"
checkpoint_dir = args.checkpoint_dir
output_dir = args.output_dir
output_name_base = args.output_name

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


# create eval dataset class
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


# custom collate function for data loading
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


# Priming
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
    out, hidden = model(inp, hidden)
    return out, hidden


def feed_sentence(model, h, sentence):
    outs = []
    for w in sentence:
        out, h = feed_input(model, h, w)
        outs.append(torch.nn.functional.log_softmax(out[0]).unsqueeze(0))
    return outs, h


# evaluation function
def eval(model, test_dataloader, init_sentence):
    condition_accuracies = defaultdict(int)
    condition_counts = defaultdict(int)
    correct_pred = 0
    sentence_details = []

    model.eval()

    hidden = model.init_hidden(1)
    init_out, init_h = feed_sentence(model, hidden, init_sentence.split(" "))

    with torch.no_grad():
        for batch in test_dataloader:
            out = None
            written = batch["sentence"]
            sentence = batch["encoded_sentence"]
            correct = batch["encoded_correct"]
            wrong = batch["encoded_wrong"]
            condition = batch["condition"]
            batch_size = sentence.size(0)
            hidden = (
                init_h[0].expand(-1, batch_size, -1).contiguous(),
                init_h[1].expand(-1, batch_size, -1).contiguous(),
            )
            for w in range(sentence.shape[1] - 1):
                word = torch.autograd.Variable(sentence[:, w].unsqueeze(0))
                out, hidden = model(word, hidden)

            log_probs = torch.nn.functional.log_softmax(out, dim=-1)
            correct_log_probs = log_probs[0, torch.arange(batch_size), correct]
            wrong_log_probs = log_probs[0, torch.arange(batch_size), wrong]
            correct_predictions = correct_log_probs >= wrong_log_probs
            for i in range(batch_size):
                cond = condition[i]
                pred = correct_predictions[i].item()
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


# Function to perform ablation
def ablate_lstm_unit(model, layer_index, unit_index):
    """Sets the weights and biases associated with a specific hidden unit in a specific layer to zero."""
    layer_str = f"lstm.weight_ih_l{layer_index}"
    if hasattr(model, "rnn"):  # For models with a single rnn layer
        layer_str = f"rnn.weight_ih_l{layer_index}"

    for name, param in model.named_parameters():
        if layer_str in name:
            param.data[:, unit_index] = 0
        elif (
            f"lstm.weight_hh_l{layer_index}" in name
            or f"rnn.weight_hh_l{layer_index}" in name
        ):
            param.data[unit_index, :] = 0
        elif (
            f"lstm.bias_ih_l{layer_index}" in name
            or f"rnn.bias_ih_l{layer_index}" in name
        ):
            start_index = unit_index
            end_index = unit_index + 4 * model.nhid
            param.data[start_index : end_index : model.nhid] = 0  # Input gate
            param.data[start_index + model.nhid : end_index : model.nhid] = (
                0  # Forget gate
            )
            param.data[start_index + 2 * model.nhid : end_index : model.nhid] = (
                0  # Cell gate
            )
            param.data[start_index + 3 * model.nhid : end_index : model.nhid] = (
                0  # Output gate
            )
        elif (
            f"lstm.bias_hh_l{layer_index}" in name
            or f"rnn.bias_hh_l{layer_index}" in name
        ):
            start_index = unit_index
            end_index = unit_index + 4 * model.nhid
            param.data[start_index : end_index : model.nhid] = 0  # Input gate
            param.data[start_index + model.nhid : end_index : model.nhid] = (
                0  # Forget gate
            )
            param.data[start_index + 2 * model.nhid : end_index : model.nhid] = (
                0  # Cell gate
            )
            param.data[start_index + 3 * model.nhid : end_index : model.nhid] = (
                0  # Output gate
            )


# Evaluate the original model
original_accuracies_list = []
for checkpoint_file in tqdm.tqdm(
    checkpoint_files, desc="Evaluating original checkpoints"
):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    model = lstm("LSTM", len(dictionary), args.emsize, args.nhid, 2, 0.2, False).to(
        device
    )
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, map_location=device)
        model.load_state_dict(state_dict)

    acc = eval(model, test_dataloader, init_sentence)
    epoch_number = get_epoch_number(checkpoint_file)
    original_accuracies_list.append({"epoch": epoch_number, "ablation": "none", **acc})

original_df = pd.DataFrame(original_accuracies_list)
output_path_original = os.path.join(output_dir, f"{output_name_base}_original.csv")
print(f"Saving original results to: {output_path_original}")
original_df.to_csv(output_path_original, index=False)
print(original_df)

# Perform ablation study for each layer and each hidden unit
for layer_index in range(2):  # Assuming 2 layers
    for unit_index in tqdm.tqdm(
        range(args.nhid), desc=f"Ablating Layer {layer_index}, LSTM units"
    ):
        ablated_accuracies_list = []
        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            model = lstm(
                "LSTM", len(dictionary), args.emsize, args.nhid, 2, 0.2, False
            ).to(device)
            with open(checkpoint_path, "rb") as f:
                state_dict = torch.load(f, map_location=device)
                model.load_state_dict(state_dict)

            # Ablate the current unit in the current layer
            ablate_lstm_unit(model, layer_index, unit_index)

            acc = eval(model, test_dataloader, init_sentence)
            epoch_number = get_epoch_number(checkpoint_file)
            ablated_accuracies_list.append(
                {
                    "epoch": epoch_number,
                    "ablation": f"layer_{layer_index}_unit_{unit_index}",
                    **acc,
                }
            )

        ablated_df = pd.DataFrame(ablated_accuracies_list)
        output_path_ablated = os.path.join(
            output_dir,
            f"{output_name_base}_ablated_layer_{layer_index}_unit_{unit_index}.csv",
        )
        print(
            f"Saving ablation results for Layer {layer_index}, Unit {unit_index} to: {output_path_ablated}"
        )
        ablated_df.to_csv(output_path_ablated, index=False)
        print(ablated_df)

print("LSTM unit ablation study complete (for both layers).")
# Example to run on CPU:
# python /scratch2/mrenaudin/colorlessgreenRNNs/evaluation_notebooks/evaluate_lstm_on_nounpp.py --emsize 650 --nhid 650 --checkpoint_dir '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/lstm_adam' --output_name 'lstm_adam_ablation'
