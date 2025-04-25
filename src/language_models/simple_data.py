import torch
from torch.utils.data import Dataset, DataLoader
import os
import logging

class TokenDataset(Dataset):
    """Simple dataset that loads tokens from a text file"""
    def __init__(self, file_path, vocab_path=None):
        self.word2idx = {}
        self.idx2word = []
        
        # Load vocabulary if provided
        if vocab_path and os.path.exists(vocab_path):
            vocab = open(vocab_path, encoding="utf8").read()
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.idx2word = vocab.split()
        else:
            # Initialize with unknown token
            self.word2idx = {"<unk>": 0}
            self.idx2word = ["<unk>"]
        
        # Load text data and tokenize
        self.tokens = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    for word in line.strip().split():
                        if word in self.word2idx:
                            self.tokens.append(self.word2idx[word])
                        else:
                            self.tokens.append(self.word2idx["<unk>"])
        
        # Convert to tensor
        self.data = torch.tensor(self.tokens, dtype=torch.long)
        logging.info(f"Loaded {len(self.data)} tokens from {file_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_vocab_size(self):
        return len(self.idx2word)


class LanguageModelDataset(Dataset):
    """Dataset for language modeling that creates sequences with targets"""
    def __init__(self, tokens, seq_length):
        """
        Args:
            tokens: Tensor of token indices
            seq_length: Length of sequences to return
        """
        self.tokens = tokens
        self.seq_length = seq_length
    
    def __len__(self):
        # Return number of possible sequences
        return max(0, len(self.tokens) - self.seq_length - 1)
    
    def __getitem__(self, idx):
        # Get input sequence starting at idx
        input_seq = self.tokens[idx:idx + self.seq_length]
        # Get target sequence (shifted by 1)
        target_seq = self.tokens[idx + 1:idx + self.seq_length + 1]
        return input_seq, target_seq


def get_batch_iterators(token_dataset, batch_size, seq_length, device=None, shuffle_train=True):
    """Creates batch iterators for train/val/test data
    
    Args:
        token_dataset: TokenDataset containing the data
        batch_size: Number of sequences per batch
        seq_length: Length of each sequence
        device: Device to move tensors to
        shuffle_train: Whether to shuffle training data
        
    Returns:
        DataLoader for creating batches with proper shaping
    """
    # Create dataset that produces sequences
    sequence_dataset = LanguageModelDataset(token_dataset.data, seq_length)
    
    # Custom collate function to ensure proper tensor shapes
    def collate_fn(batch):
        # Separate inputs and targets
        inputs, targets = zip(*batch)
        
        # Stack along batch dimension
        inputs = torch.stack(inputs)  # Shape: [batch_size, seq_len]
        targets = torch.stack(targets)  # Shape: [batch_size, seq_len]
        
        # Transpose to match expected format [seq_len, batch_size]
        inputs = inputs.transpose(0, 1)
        
        # Flatten targets for loss calculation
        targets = targets.reshape(-1)
        
        # Move to device if specified
        if device is not None:
            inputs = inputs.to(device)
            targets = targets.to(device)
        
        return inputs, targets
    
    # Create DataLoader with collate_fn
    dataloader = DataLoader(
        sequence_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=True  # Drop last incomplete batch
    )
    
    return dataloader 