import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import csv
from src.language_models.model import RNNModel
from src.language_models.utils import move_to_device, batchify, get_batch
from src.language_models.dictionary_corpus import Dictionary, tokenize

parser = argparse.ArgumentParser(description='NounPP evaluation')

parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")

eval_batch_size = 1
total_correct = 0
total_count = 0

dictionary = Dictionary("/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli")
vocab_size = len(dictionary)
eval_batch_size = 1
seq_len=6
tokens = tokenize(dictionary, "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.text")

test_data = batchify(tokens, eval_batch_size, device)

nounpp_txt = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt"
nounpp_gold = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.gold"

with open(nounpp_txt, "r", encoding="utf8") as f:
    lines_txt = [line.strip().split() for line in f.readlines()]
    conditions = [(line[7], line[8]) for line in lines_txt]  # Extract conditions
with open(nounpp_gold, "r", encoding="utf8") as f:
    lines_gold = [line.strip().split("\t") for line in f.readlines()]
    verb_pairs = [(line[1], line[2]) for line in lines_gold]  # Extract correct & incorrect verbs

print('conditions:', len(conditions), conditions[0])
print('verb_pairs:', len(verb_pairs), verb_pairs[0])

# Reshape the test_data into (num_sentences, 6) where last word is the target
num_sentences = test_data.size(0) // seq_len
test_data = test_data.view(num_sentences, seq_len).to(device)
model = RNNModel('LSTM', 50001, 200, 650, 2, 0.2, False)
model = model.to(device)
with open("/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/full_check/epoch_10.pt", 'rb') as f:
      print("Loading the model")
      state_dict = torch.load(f, map_location='cuda' if args.cuda else 'cpu')
      model.load_state_dict(state_dict)
model =model.to(device)
model.eval()  # Set model to evaluation mode
condition_correct_counts = {}  # Tracks correct predictions per condition
condition_total_counts = {} 
with torch.no_grad():  # Disable gradient computation for evaluation
    for i in range(num_sentences):
        sentence = test_data[i, :-1].unsqueeze(1)  
        sentence = sentence.long().to(device)
        
        # Get correct & incorrect verb from verb_pairs
        correct_verb, incorrect_verb = verb_pairs[i]
        # Convert words to token IDs
        correct_verb_id = torch.tensor(dictionary.word2idx[correct_verb]).to(device)  
        incorrect_verb_id = torch.tensor(dictionary.word2idx[incorrect_verb]).to(device)  
        condition = conditions[i]
        # Mask last word and run model
        hidden = move_to_device(model.init_hidden(eval_batch_size), device)
        

        output, hidden = model(sentence, hidden)  # Forward pass
        
        # Extract logits for the last word position
        output_probs = torch.softmax(output[:, -1, :], dim=-1).to(device)  # Convert logits to probabilities
        # Compare probabilities for correct & incorrect verbs
        prob_correct = output_probs[0, correct_verb_id].item()
        prob_incorrect = output_probs[0, incorrect_verb_id].item()
        # Check if model prefers the correct verb
        is_correct = prob_correct >= prob_incorrect

        # Update condition statistics
        if condition not in condition_correct_counts:
            condition_correct_counts[condition] = 0
            condition_total_counts[condition] = 0

        condition_correct_counts[condition] += int(is_correct)
        condition_total_counts[condition] += 1
        total_correct += int(is_correct)
        total_count += 1
print("\n=== Average Accuracy per Condition ===")
for condition in condition_correct_counts:
    accuracy = condition_correct_counts[condition] / condition_total_counts[condition]
    print(f"Condition {condition}: {accuracy:.2%}")
total_accuracy = total_correct / total_count
print(f"\nTotal Accuracy: {total_accuracy:.2%}")