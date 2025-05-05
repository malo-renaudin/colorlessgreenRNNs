# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
from collections import defaultdict
import logging
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random




class Dictionary(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = defaultdict(int)

        vocab_path = os.path.join(path, "vocab.txt")
        try:
            vocab = open(vocab_path, encoding="utf8").read()
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.idx2word = [w for w in vocab.split()]
            self.vocab_file_exists = True
        except FileNotFoundError:
            logging.info("Vocab file not found, creating new vocab file.")
            self.create_vocab(os.path.join(path, "nounpp.txt"))
            open(vocab_path, "w").write("\n".join([w for w in self.idx2word]))

    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        # return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self, path):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.add_word(word)


# class Corpus(object):
#     def __init__(self, path):
#         self.dictionary = Dictionary(path)
#         self.train = tokenize(self.dictionary, os.path.join(path, 'train.txt'))
#         self.valid = tokenize(self.dictionary, os.path.join(path, 'valid.txt'))
#         self.test = tokenize(self.dictionary, os.path.join(path, 'test.txt'))
class Corpus(object):
    def __init__(self, path, save_tokenized=True):  # added save_tokenized parameter.
        self.dictionary = Dictionary(path)
        self.train_path = os.path.join(path, "train.txt")
        self.valid_path = os.path.join(path, "valid.txt")
        self.test_path = os.path.join(path, "test.txt")

        if save_tokenized:
            self.train = self.tokenize_and_save(self.train_path, "train_tokenized.pkl")
            self.valid = self.tokenize_and_save(self.valid_path, "valid_tokenized.pkl")
            self.test = self.tokenize_and_save(self.test_path, "test_tokenized.pkl")
        else:
            self.train = self.load_tokenized("train_tokenized.pkl", self.train_path)
            self.valid = self.load_tokenized("valid_tokenized.pkl", self.valid_path)
            self.test = self.load_tokenized("test_tokenized.pkl", self.test_path)

    def tokenize_and_save(self, path, save_path):
        """Tokenizes a text file and saves the tokenized data."""
        tokenized_data = tokenize(self.dictionary, path)
        with open(save_path, "wb") as f:
            pickle.dump(tokenized_data, f)
        return tokenized_data

    def load_tokenized(self, save_path, original_path):
        """loads a tokenized file, or tokenizes and saves it if it does not exist."""
        try:
            with open(save_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            logging.info(f"{save_path} not found, creating tokenized data.")
            return self.tokenize_and_save(original_path, save_path)


def tokenize(dictionary, path, shuffle=False):
    """Tokenizes a text file to a sequence of indices format.
       Assumes that training and test data have <eos> symbols.
    """
    assert os.path.exists(path)

    # Read all lines
    with open(path, 'r', encoding="utf8") as f:
        lines = f.readlines()

    if shuffle:
        random.shuffle(lines)

    # Count total number of tokens
    ntokens = 0
    for line in lines:
        words = line.split()
        ntokens += len(words)

    # Allocate tensor
    ids = torch.LongTensor(ntokens)

    # Fill tensor
    token = 0
    for line in lines:
        words = line.split()
        for word in words:
            if word in dictionary.word2idx:
                ids[token] = dictionary.word2idx[word]
            else:
                ids[token] = dictionary.word2idx.get("<unk>", 0)
            token += 1

    return ids



########################################################
# New implementation with padding and masking and shuffling by sentence
########################################################

class Vocabulary:
    def __init__(self, filepath=None, add_special_tokens=True):
        self.word2idx = {}
        self.idx2word = []
        self.special_tokens = []

        if add_special_tokens:
            self.add_special_token("<pad>") 
            self.add_special_token("<unk>")

        if filepath:
            self.load_vocab(filepath)

    def add_special_token(self, token):
        if token not in self.word2idx:
            self.idx2word.append(token)
            self.word2idx[token] = len(self.word2idx)
            self.special_tokens.append(token)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.word2idx)

    def load_vocab(self, filepath):
        print(f"Loading vocabulary from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for word in f:
                    word = word.strip()
                    if word: # Avoid empty lines
                        self.add_word(word)
            print(f"Vocabulary loaded. Size: {len(self)}")
        except FileNotFoundError:
            print(f"Error: Vocabulary file not found at {filepath}")
            raise

    def __len__(self):
        return len(self.idx2word)

    def get_index(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def get_word(self, index):
        if 0 <= index < len(self.idx2word):
            return self.idx2word[index]
        return "<unk>" # Or raise an error

    @property
    def pad_idx(self):
        return self.word2idx["<pad>"]

    @property
    def unk_idx(self):
        return self.word2idx["<unk>"]
    
def word_tokenizer(text):
    return text.split()

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, vocab):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.sentences_as_indices = []

        print(f"Loading and processing data from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    sentence = line.strip()
                    if sentence: 
                        tokens = self.tokenizer(sentence)
                        indices = [self.vocab.get_index(token) for token in tokens]
                        self.sentences_as_indices.append(torch.tensor(indices, dtype=torch.long))
            print(f"Loaded {len(self.sentences_as_indices)} sentences.")
        except FileNotFoundError:
            print(f"Error: Data file not found at {filepath}")
            raise 

    def __len__(self):
        return len(self.sentences_as_indices)

    def __getitem__(self, idx):
        return self.sentences_as_indices[idx]
    
def collate_batch(batch, pad_idx):
    """
    Collates sequences of varying lengths into a batch.

    Args:
        batch (list): A list of tensors, where each tensor is a sequence of indices.
        pad_idx (int): The index used for padding.

    Returns:
        tuple:
            - padded_sequences (torch.Tensor): Batch of sequences, padded to the
              length of the longest sequence in the batch (batch_size, max_seq_len).
            - padding_mask (torch.Tensor): Boolean tensor indicating padding positions.
              'True' for non-pad tokens, 'False' for pad tokens (batch_size, max_seq_len).
              This mask format is often useful for masking loss or in attention.
    """


    # Pad the input sequences
    input_sequences = pad_sequence(batch, batch_first=True, padding_value=pad_idx)
    batch_size, max_seq_len = input_sequences.shape


    # Create the target sequences by shifting the inputs left
    # Target for input token at index `t` is the token at `t+1`
    # Slice to remove the first token of each sequence
    target_sequences_shifted = input_sequences[:, 1:]

    # Create a tensor of pad tokens to append to the end of each target sequence
    # Shape: (batch_size, 1)
    pad_tensor = torch.full((batch_size, 1), pad_idx, dtype=torch.long, device=input_sequences.device)

    # Concatenate the shifted sequence with the padding tensor
    # Now target_sequences has the same shape as input_sequences
    target_sequences = torch.cat([target_sequences_shifted, pad_tensor], dim=1)

    return input_sequences, target_sequences