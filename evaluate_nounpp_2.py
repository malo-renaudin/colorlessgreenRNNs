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
training_data = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"
dictionary = Dictionary(training_data)
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
with open("/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/full_check/epoch_40.pt", 'rb') as f:
    print("Loading the model")
    #uncomment with my checkpointing, here try with their checkpointing to see if results match
    state_dict = torch.load(f, map_location='cuda' if args.cuda else 'cpu')
    model.load_state_dict(state_dict)

    # if args.cuda:
    #     model = torch.load(f)
    # else:
    #     # to convert model trained on cuda to cpu model
    #     model = torch.load(f, map_location = lambda storage, loc: storage)
model.eval()  # Set model to evaluation mode
model = model.to(device)
condition_correct_counts = {}  # Tracks correct predictions per condition
condition_total_counts = {} 
with torch.no_grad():  # Disable gradient computation for evaluation
    for i in range(num_sentences):
        sentence = test_data[i, :-1].unsqueeze(1)  
        sentence = sentence.long().to(device)
        
        # Get correct & incorrect verb from verb_pairs
        correct_verb, incorrect_verb = verb_pairs[i]
        # Convert words to token IDs
        if correct_verb in dictionary.word2idx and incorrect_verb in dictionary.word2idx:

            correct_verb_id = torch.tensor(dictionary.word2idx[correct_verb]).to(device)  
            incorrect_verb_id = torch.tensor(dictionary.word2idx[incorrect_verb]).to(device)  
        else : 
            print('pb')
            continue
        condition = conditions[i]
        # Mask last word and run model
        hidden = move_to_device(model.init_hidden(eval_batch_size), device)
        

        output, hidden = model(sentence, hidden)  # Forward pass
        
        # Extract logits for the last word position
        output_probs = F.log_softmax(output[:, -1, :], dim=-1).to(device)  # Convert logits to probabilities
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