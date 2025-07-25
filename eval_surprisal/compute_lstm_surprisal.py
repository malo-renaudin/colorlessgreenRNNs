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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
import sys
import os
from pathlib import Path

# # Add path for your modules
# sys.path.insert(0, "./colorlessgreenRNNs/src/language_models")
from language_models.dictionary_corpus import Dictionary
from language_models.model import RNNModel

# Suppress warnings
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

class SurprisalComputer:
    def __init__(self, model_path, data_path, device=None):
        """
        Initialize surprisal computer
        
        Args:
            model_path: Path to the trained model checkpoint
            data_path: Path to the data directory containing dictionary
            device: torch device to use
        """
        self.model_path = model_path
        self.data_path = data_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load dictionary
        self.dictionary = Dictionary(data_path)
        
        # Load and setup model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the RNN model from checkpoint"""
        # Create model with same architecture as training
        model = RNNModel(
            rnn_type="LSTM", 
            ntoken=len(self.dictionary),  # Use actual vocab size from dictionary
            ninp=650,      # embedding dimension
            nhid=650,      # hidden dimension  
            nlayers=2,     # number of layers
            dropout=0.2,   # dropout rate
            tie_weights=False
        )
        
        print(f"Model architecture: {model.rnn_type}, vocab_size={len(self.dictionary)}, "
              f"emb_dim={model.encoder.embedding_dim}, hidden_dim={model.nhid}, "
              f"layers={model.nlayers}")
        
        # Load state dict
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict['model_state_dict'])
            print(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def tokenize(self, sentence):
        """
        Tokenize sentence using the same logic as original code
        """
        sent = sentence.strip()
        if sent == "":
            return []
        
        # Respect commas as tokens
        sent = " ,".join(sent.split(","))
        
        # Handle end-of-sentence punctuation
        if sent[-1] in [".", "?", "!"]:
            sent = sent[:-1] + " " + sent[-1]
        
        # Check for periods in middle of sentence
        if ("." in sent) and (sent[-1] != "."):
            print(f"Warning: period in middle of sentence: {sent}")
        
        # Split on contractions
        sent = " 's".join(sent.split("'s"))
        sent = " n't".join(sent.split("n't"))
        
        return sent.split()
    
    def indexify(self, word):
        """Convert word to vocabulary index"""
        if word not in self.dictionary.word2idx:
            print(f"Warning: {word} not in vocab")
            return self.dictionary.word2idx.get("<unk>", 0)
        return self.dictionary.word2idx[word]
    
    def compute_sentence_surprisals(self, sentence, uncased=False):
        """
        Compute surprisal for each word in a sentence
        
        Args:
            sentence: Input sentence string
            uncased: Whether to lowercase words
            
        Returns:
            List of dictionaries with word-level surprisal information
        """
        # Tokenize sentence
        tokens = self.tokenize(sentence)
        if not tokens:
            return []
        
        # Add beginning-of-sentence token
        sentence_tokens = ["<eos>"] + tokens
        
        # Convert to indices
        indices = []
        for word in sentence_tokens:
            word_to_index = word.lower() if uncased else word
            idx = self.indexify(word_to_index)
            indices.append(idx)
        
        # Convert to tensor
        input_tensor = torch.LongTensor(indices).to(self.device)
        
        # Run forward pass
        with torch.no_grad():
            hidden = self.model.init_hidden(1)
            output, _ = self.model(input_tensor.view(-1, 1), hidden)
        
        # Compute surprisals
        results = []
        for i, (word_idx, word) in enumerate(zip(indices[1:], tokens)):  # Skip BOS token
            # Get log probabilities
            log_probs = F.log_softmax(output[i], dim=-1).view(-1)
            
            # Surprisal = -log P(word|context)
            surprisal = -log_probs[word_idx].item()
            surprisal_bits = surprisal / np.log(2.0)  # Convert to bits
            
            # Create result dictionary
            result = {
                'word': word,
                'WordPosition': i,  # 0-indexed position
                'token': word if word in self.dictionary.word2idx else "<UNK>",
                'surprisal': surprisal,
                'surprisal_bits': surprisal_bits,
                'log_prob': log_probs[word_idx].item()
            }
            results.append(result)
        
        return results
    
    def process_dataset(self, dataset_path, output_path, uncased=False):
        """
        Process a dataset file and compute surprisals
        
        Args:
            dataset_path: Path to input dataset (one sentence per line with index)
            output_path: Path to save output CSV
            uncased: Whether to use uncased processing
        """
        print(f"Processing dataset: {dataset_path}")
        print(f"Output will be saved to: {output_path}")
        
        results = []
        
        # Read dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line_num == 0:
                    continue
                print(line)

                # Parse line - assuming format: "index sentence"
                parts = line.split(' ', 1)
                if len(parts) < 2:
                    print(f"Warning: Skipping malformed line {line_num}: {line}")
                    continue
                
                idx, sentence = parts
                print(sentence)
                try:
                    # Compute surprisals for this sentence
                    sentence_results = self.compute_sentence_surprisals(sentence, uncased=uncased)
                    
                    # Add sentence metadata to each word result
                    for word_result in sentence_results:
                        word_result['sentence_id'] = int(idx)
                        word_result['sentence'] = sentence
                        results.append(word_result)
                    
                    if (line_num + 1) % 100 == 0:
                        print(f"Processed {line_num + 1} sentences...")
                        
                except Exception as e:
                    print(f"Error processing sentence {idx}: {e}")
                    continue
        
        # Convert to DataFrame and save
        df = pd.DataFrame(results)
        
        # Reorder columns for clarity
        column_order = ['sentence_id', 'sentence', 'word_pos', 'word', 'token', 
                       'surprisal', 'surprisal_bits', 'log_prob']
        df = df[column_order]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} word-level surprisal measurements to {output_path}")
        
        return df
    
    def process_csv_format(self, input_csv, output_csv, sentence_column='Sentence', uncased=False):
        """
        Process CSV file with sentences and compute surprisals
        
        Args:
            input_csv: Path to input CSV file
            output_csv: Path to output CSV file  
            sentence_column: Name of column containing sentences
            uncased: Whether to use uncased processing
        """
        print(f"Processing CSV: {input_csv}")
        
        # Read input CSV
        df_input = pd.read_csv(input_csv)
        
        if sentence_column not in df_input.columns:
            raise ValueError(f"Column '{sentence_column}' not found in input CSV")
        
        all_results = []
        
        for idx, row in df_input.iterrows():
            sentence = row[sentence_column]
            
            try:
                # Compute surprisals
                word_results = self.compute_sentence_surprisals(sentence, uncased=uncased)
                
                # Add original row data to each word result
                for word_result in word_results:
                    # Copy all original columns
                    for col, val in row.items():
                        word_result[col] = val
                    
                    all_results.append(word_result)
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Save results
        df_output = pd.DataFrame(all_results)
        df_output.to_csv(output_csv, index=False)
        print(f"Saved surprisal results to {output_csv}")
        
        return df_output

def main():
    """Main function to run surprisal computation"""
    
    # Configuration
    model_path = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/lstm_adam_full_check_shuffled/epoch_40.pt"
    data_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"
    
    # Paths to your datasets
    train_path = "/scratch2/mrenaudin/colorlessgreenRNNs/eval_surprisal/sentences/Fillers.csv"  # Update with actual path
    test_path = "/scratch2/mrenaudin/colorlessgreenRNNs/eval_surprisal/sentences/items_filler.pivot.csv"    # Update with actual path
    
    # Output paths
    train_output = "train_surprisals.csv"
    test_output = "test_surprisals.csv"
    
    # Initialize surprisal computer
    print("Initializing surprisal computer...")
    computer = SurprisalComputer(model_path, data_path)
    
    # Process train set
    if os.path.exists(train_path):
        print("\nProcessing training set...")
        train_df = computer.process_dataset(train_path, train_output, uncased=False)
        print(f"Training set: {len(train_df)} word measurements")
    else:
        print(f"Training file not found: {train_path}")
    
    # Process test set  
    if os.path.exists(test_path):
        print("\nProcessing test set...")
        test_df = computer.process_dataset(test_path, test_output, uncased=False)
        print(f"Test set: {len(test_df)} word measurements")
    else:
        print(f"Test file not found: {test_path}")
    
    print("\nSurprisal computation completed!")

if __name__ == "__main__":
    main()