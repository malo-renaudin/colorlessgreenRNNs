import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import datasets
from collections import Counter, defaultdict
import pickle
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

class WordTokenizer:
    """Classic word-level tokenizer for WikiText-103"""
    
    def __init__(self, vocab_size: int = 50000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.BOS_TOKEN = '<bos>'
        self.EOS_TOKEN = '<eos>'
        
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        
    def _tokenize_text(self, text: str) -> List[str]:
        """Basic word tokenization with punctuation handling"""
        # Convert to lowercase and handle basic punctuation
        text = text.lower()
        # Split on whitespace and basic punctuation
        text = re.sub(r'([.!?;,:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = text.strip().split()
        return tokens
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from training texts"""
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in texts:
            tokens = self._tokenize_text(text)
            self.word_counts.update(tokens)
        
        print(f"Total unique words before filtering: {len(self.word_counts)}")
        
        # Filter by minimum frequency and take top vocab_size - special_tokens
        filtered_words = [word for word, count in self.word_counts.items() 
                         if count >= self.min_freq]
        
        # Sort by frequency and take top words
        most_common = self.word_counts.most_common()
        vocab_words = [word for word, count in most_common 
                      if count >= self.min_freq][:self.vocab_size - len(self.special_tokens)]
        
        # Build word-to-id mapping
        self.word2idx = {}
        self.idx2word = {}
        
        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token
        
        # Add vocabulary words
        for i, word in enumerate(vocab_words):
            idx = i + len(self.special_tokens)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Final vocabulary size: {len(self.word2idx)}")
        print(f"Most common words: {vocab_words[:10]}")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = self._tokenize_text(text)
        token_ids = []
        
        for token in tokens:
            if token in self.word2idx:
                token_ids.append(self.word2idx[token])
            else:
                token_ids.append(self.word2idx[self.UNK_TOKEN])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.idx2word:
                tokens.append(self.idx2word[token_id])
            else:
                tokens.append(self.UNK_TOKEN)
        
        return ' '.join(tokens)
    
    def save(self, filepath: str) -> None:
        """Save tokenizer to file"""
        tokenizer_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.word2idx = tokenizer_data['word2idx']
        self.idx2word = tokenizer_data['idx2word']
        self.vocab_size = tokenizer_data['vocab_size']
        self.min_freq = tokenizer_data['min_freq']
        self.special_tokens = tokenizer_data['special_tokens']
        print(f"Tokenizer loaded from {filepath}")
    
    @property
    def pad_token_id(self) -> int:
        return self.word2idx[self.PAD_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        return self.word2idx[self.UNK_TOKEN]
    
    @property
    def bos_token_id(self) -> int:
        return self.word2idx[self.BOS_TOKEN]
    
    @property
    def eos_token_id(self) -> int:
        return self.word2idx[self.EOS_TOKEN]


class WikiTextDataset(Dataset):
    """PyTorch Dataset for WikiText-103 with word tokenization, compatible with CBR_RNN"""
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer: WordTokenizer, 
        sequence_length: int = 35,  # Changed from max_length to sequence_length to match CBR_RNN
        add_special_tokens: bool = False,  # CBR_RNN typically doesn't use special tokens
        create_sequences: bool = True  # Create overlapping sequences for language modeling
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.add_special_tokens = add_special_tokens
        self.create_sequences = create_sequences
        
        # Pre-tokenize all texts and create sequences
        print("Pre-tokenizing texts and creating sequences...")
        self.sequences = []
        
        for text in texts:
            if text.strip():  # Skip empty texts
                tokens = self.tokenizer.encode(text)
                if self.add_special_tokens:
                    tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
                
                if self.create_sequences:
                    # Create overlapping sequences of length sequence_length + 1
                    # +1 because we need input and target (shifted by 1)
                    for i in range(0, len(tokens) - self.sequence_length, self.sequence_length):
                        if i + self.sequence_length + 1 <= len(tokens):
                            sequence = tokens[i:i + self.sequence_length + 1]
                            self.sequences.append(sequence)
                else:
                    # Just store the tokens as-is, truncated to sequence_length + 1
                    if len(tokens) >= self.sequence_length + 1:
                        self.sequences.append(tokens[:self.sequence_length + 1])
        
        print(f"Dataset created with {len(self.sequences)} sequences of length {self.sequence_length}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input, target) tensors for language modeling"""
        sequence = self.sequences[idx]
        
        # Input is all tokens except the last one
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        # Target is all tokens except the first one (shifted by 1)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids


def collate_fn_cbr(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for CBR_RNN - expects (sequence_length, batch_size) format"""
    inputs, targets = zip(*batch)
    
    # Stack inputs and targets
    input_batch = torch.stack(inputs)    # (batch_size, sequence_length)
    target_batch = torch.stack(targets)  # (batch_size, sequence_length)
    
    # Transpose to match CBR_RNN expected format: (sequence_length, batch_size)
    input_batch = input_batch.transpose(0, 1)
    target_batch = target_batch.transpose(0, 1)
    
    return input_batch, target_batch


class WikiTextDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for WikiText-103, compatible with CBR_RNN"""
    
    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 50000,
        min_freq: int = 2,
        sequence_length: int = 35,  # Changed from max_length to sequence_length
        batch_size: int = 32,
        num_workers: int = 4,
        add_special_tokens: bool = False,  # CBR_RNN typically doesn't use special tokens
        create_sequences: bool = True
    ):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.add_special_tokens = add_special_tokens
        self.create_sequences = create_sequences
        
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self) -> None:
        """Check if local WikiText-103 dataset exists"""
        data_path = Path(self.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Local dataset not found at {self.data_path}")
        
        required_splits = ['train', 'validation', 'test']
        for split in required_splits:
            split_path = data_path / split
            if not split_path.exists():
                raise FileNotFoundError(f"Split '{split}' not found at {split_path}")
        
        print(f"Using local WikiText-103 dataset from {self.data_path}")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets and tokenizer"""
        
        # Load local dataset
        print(f"Loading local dataset from {self.data_path}")
        wikitext = datasets.load_from_disk(self.data_path)
        
        # Initialize or load tokenizer
        if self.tokenizer_path and Path(self.tokenizer_path).exists():
            self.tokenizer = WordTokenizer()
            self.tokenizer.load(self.tokenizer_path)
        else:
            self.tokenizer = WordTokenizer(
                vocab_size=self.vocab_size, 
                min_freq=self.min_freq
            )
            
            # Build vocabulary on training data
            train_texts = [text for text in wikitext['train']['text'] if text.strip()]
            self.tokenizer.build_vocab(train_texts)
            
            # Save tokenizer if path provided
            if self.tokenizer_path:
                self.tokenizer.save(self.tokenizer_path)
        
        # Create datasets
        if stage == "fit" or stage is None:
            train_texts = [text for text in wikitext['train']['text'] if text.strip()]
            val_texts = [text for text in wikitext['validation']['text'] if text.strip()]
            
            self.train_dataset = WikiTextDataset(
                train_texts, self.tokenizer, self.seq_len, 
                self.add_special_tokens, self.create_sequences
            )
            self.val_dataset = WikiTextDataset(
                val_texts, self.tokenizer, self.seq_len,
                self.add_special_tokens, self.create_sequences
            )
        
        if stage == "test" or stage is None:
            test_texts = [text for text in wikitext['test']['text'] if text.strip()]
            self.test_dataset = WikiTextDataset(
                test_texts, self.tokenizer, self.seq_len,
                self.add_special_tokens, self.create_sequences
            )
    
    def prepare_data(self) -> None:
        """Download WikiText-103 dataset"""
        print("Loading WikiText-103 dataset...")
        datasets.load_dataset("wikitext", "wikitext-103-v1")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets and tokenizer"""
        
        # Load dataset
        #wikitext = datasets.load_dataset("wikitext", "wikitext-103-v1")
        
        # Initialize or load tokenizer
        if self.tokenizer_path and Path(self.tokenizer_path).exists():
            self.tokenizer = WordTokenizer()
            self.tokenizer.load(self.tokenizer_path)
        else:
            self.tokenizer = WordTokenizer(
                vocab_size=self.vocab_size, 
                min_freq=self.min_freq
            )
            
            # Build vocabulary on training data
            train_texts = [text for text in wikitext['train']['text'] if text.strip()]
            self.tokenizer.build_vocab(train_texts)
            
            # Save tokenizer if path provided
            if self.tokenizer_path:
                self.tokenizer.save(self.tokenizer_path)
        
        # Create datasets
        if stage == "fit" or stage is None:
            train_texts = [text for text in wikitext['train']['text'] if text.strip()]
            val_texts = [text for text in wikitext['validation']['text'] if text.strip()]
            
            self.train_dataset = WikiTextDataset(
                train_texts, self.tokenizer, self.sequence_length, 
                self.add_special_tokens, self.create_sequences
            )
            self.val_dataset = WikiTextDataset(
                val_texts, self.tokenizer, self.sequence_length,
                self.add_special_tokens, self.create_sequences
            )
        
        if stage == "test" or stage is None:
            test_texts = [text for text in wikitext['test']['text'] if text.strip()]
            self.test_dataset = WikiTextDataset(
                test_texts, self.tokenizer, self.sequence_length,
                self.add_special_tokens, self.create_sequences
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn_cbr,
            drop_last=True  # Ensure consistent batch sizes for CBR_RNN
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_cbr,
            drop_last=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_cbr,
            drop_last=True
        )
