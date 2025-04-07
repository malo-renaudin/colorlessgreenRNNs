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


def tokenize(dictionary, path_or_sentence, is_path=True):
    """Tokenizes a text file for training or testing to a sequence of indices format
    We assume that training and test data has <eos> symbols"""
    # assert os.path.exists(path)
    # with open(path, 'r', encoding="utf8") as f:
    #     ntokens = 0
    #     for line in f:
    #         words = line.split()
    #         ntokens += len(words)
    if is_path:
        assert os.path.exists(path_or_sentence)
        with open(path_or_sentence, "r", encoding="utf8") as f:
            ntokens = 0
            for line in f:
                words = line.split()
                ntokens += len(words)
        # Tokenize file content
        with open(path_or_sentence, "r", encoding="utf8") as f:
            ids = torch.LongTensor(ntokens)
            token = 0
            for line in f:
                words = line.split()
                for word in words:
                    if word in dictionary.word2idx:
                        ids[token] = dictionary.word2idx[word]
                    else:
                        ids[token] = dictionary.word2idx["<unk>"]
                    token += 1

        return ids
    else:
        # Tokenize a list of sentences
        all_tokens = []
        for sentence in path_or_sentence:  # Process each sentence in the list
            words = sentence.split()  # Split sentence into words
            ntokens = len(words)
            ids = torch.LongTensor(ntokens)
            for token, word in enumerate(words):
                if word in dictionary.word2idx:
                    ids[token] = dictionary.word2idx[word]
                else:
                    ids[token] = dictionary.word2idx["<unk>"]
            all_tokens.append(ids)

        return all_tokens
