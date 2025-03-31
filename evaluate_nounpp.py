import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import csv
from src.language_models.model import RNNModel
from src.language_models.utils import move_to_device
parser = argparse.ArgumentParser(description='NounPP evaluation')

parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")

def word_tokenize(vocab_path, sentence):
    # Load vocabulary file
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = f.read().splitlines()
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}  # Create word to index mapping
    print(word2idx)
    unk_token = "<unk>"
    unk_idx = len(vocab)  # Index for <unk> token

    # Tokenize sentences and map words to indices
    tokenized_sentence = []
    tokens = sentence.split()  # Split sentence into words
    token_ids = [word2idx.get(word, unk_idx) for word in tokens]  # Map words to indices
    tokenized_sentence.append(token_ids)

    return tokenized_sentence
# Load LSTM Model
def load_model(model_path, device):
    with open(model_path, 'rb') as f:
        print("Loading the model")
        state_dict = torch.load(f, map_location='cuda' if args.cuda else 'cpu')
        model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


# Evaluate the model on the NounPP dataset
def evaluate_nounpp(model, eval_batch_size, vocab_path, nounpp_txt, nounpp_gold, output_file):
    # Read sentences from Nounpp.txt
    with open(nounpp_txt, "r", encoding="utf8") as f:
        lines_txt = [line.strip() for line in f.readlines()]
        sentences = [line[1] for line in lines_txt]
        conditions = [(line[2], line[3]) for line in lines_txt]
        print('sentences', len(sentences))
        print(sentences[0])
        print('conditions',len(conditions))
        print(conditions[0])
    
    # Read correct & incorrect verbs + conditions from Nounpp.gold
    with open(nounpp_gold, "r", encoding="utf8") as f:
        lines_gold = [line.strip().split("\t") for line in f.readlines()]
        verb_pairs = [(line[1], line[2]) for line in lines_gold]  # Extract correct & incorrect verbs
        print("verb_pairs", len(verb_pairs))
        print(verb_pairs[0])
    # Initialize result storage
    results = {cond: [] for cond in set(conditions)}

    for sentence, (correct_verb, incorrect_verb), condition in zip(sentences, verb_pairs, conditions):
        tokens = word_tokenize(vocab_path, sentence)
        print('tokens', tokens)
        if not tokens:
            continue

        # Identify verb position (last word)
        verb_pos = len(tokens) - 1

        # Mask all words except the verb
        masked_tokens = torch.LongTensor(tokens).unsqueeze(0)  # Shape: (1, sequence_length)
        hidden = move_to_device(model.init_hidden(eval_batch_size), device)
        # Get model predictions (logits)
        with torch.no_grad():
            outputs, _ = model(masked_tokens, hidden)  # Output shape: (1, seq_len, vocab_size)
            logits = outputs[0, verb_pos]  # Extract logits at the verb position

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Get probabilities of correct and incorrect verbs
        correct_idx = word_tokenize(vocab_path, [correct_verb])[0][0]
        incorrect_idx = word_tokenize(vocab_path, [incorrect_verb])[0][0]

        correct_prob = probs[correct_idx].item()
        incorrect_prob = probs[incorrect_idx].item()

        # Store result (1 if correct_prob > incorrect_prob, else 0)
        results[condition].append(1 if correct_prob > incorrect_prob else 0)

    # Compute and save average accuracy per condition
    avg_accuracies = {condition: np.mean(scores) if scores else 0 for condition, scores in results.items()}
    return avg_accuracies


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Paths
    checkpoint_dir = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/full_check"
    vocab_path = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/vocab.txt"  # Path to dictionary
    nounpp_txt = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt"
    nounpp_gold = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.gold"
    output_dir = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/results"
    output_file = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/results/results.csv"
    

    os.makedirs(output_dir, exist_ok=True)
    
    model = RNNModel('LSTM', 50001, 200, 650, 2, 0.2, False)
    model = model.to(device)
    eval_batch_size = 1
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)

        # Iterate over checkpoints
        for checkpoint_file in os.listdir(checkpoint_dir):
            if checkpoint_file.endswith('.pt') or checkpoint_file.endswith('.pth'):  # Filter only model files
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
                model = load_model(checkpoint_path, device)
                
                # Get the checkpoint name (without extension)
                checkpoint_name = os.path.splitext(checkpoint_file)[0]
                
                # Evaluate the model
                avg_accuracies = evaluate_nounpp(model, eval_batch_size, vocab_path, nounpp_txt, nounpp_gold, output_file)
                condition_names = list(avg_accuracies.keys())  # Dynamically extract conditions from results
                if f.tell() == 0:  # Check if it's the first row (file is empty)
                    writer.writerow(["Checkpoint"] + condition_names)  # Write header row

                # Write the checkpoint name and average accuracies for each condition
                row = [checkpoint_name] + [avg_accuracies.get(cond, 0) for cond in condition_names]  # Write accuracy for each condition
                writer.writerow(row)
            print("checkpoint processed")
    
    print(f"Results saved to {output_file}")