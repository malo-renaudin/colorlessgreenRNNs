import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import datasets
import re
from collections import Counter
import pickle
import os

class SimpleWordTokenizer:
    """Simple word-level tokenizer for WikiText"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        # Special tokens - only UNK needed
        self.UNK_TOKEN = '<unk>'
        
    def _tokenize_text(self, text):
        """Simple tokenization: lowercase, split on whitespace and punctuation"""
        # Convert to lowercase and split on whitespace/punctuation
        text = text.lower()
        # Keep apostrophes in contractions, split on other punctuation
        words = re.findall(r"\b\w+(?:'\w+)?\b|[.!?;]", text)
        return words
    
    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from list of texts"""
        print("Building vocabulary...")
        
        # Count all words
        for text in texts:
            if text.strip():  # Skip empty lines
                words = self._tokenize_text(text)
                self.word_counts.update(words)
        
        print(f"Found {len(self.word_counts)} unique words")
        
        # Create vocabulary with only UNK token
        vocab = [self.UNK_TOKEN]
        
        # Add most frequent words
        most_common = self.word_counts.most_common(self.vocab_size - len(vocab))
        for word, count in most_common:
            if count >= min_freq:
                vocab.append(word)
        
        # Create mappings
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Built vocabulary with {len(self.word2idx)} words")
        print(f"Most common words: {list(self.word2idx.keys())[1:11]}")  # Skip UNK token
        
    def encode(self, text):
        """Convert text to list of token indices"""
        words = self._tokenize_text(text)
        return [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]
    
    def decode(self, token_ids):
        """Convert list of token indices back to text"""
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in token_ids]
        return ' '.join(words)
    
    def save(self, filepath):
        """Save tokenizer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, filepath):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_counts = data['word_counts']
            self.vocab_size = data['vocab_size']

class WikiTextDataset(Dataset):
    """Simple WikiText dataset - cuts sequences at exact length, no padding"""
    
    def __init__(self, token_ids, sequence_length=128):
        self.token_ids = token_ids
        self.sequence_length = sequence_length
        
        # Calculate number of complete sequences we can extract
        self.num_sequences = len(token_ids) // sequence_length
        
        # Truncate to exact multiple of sequence_length
        self.token_ids = token_ids[:self.num_sequences * sequence_length]
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # Calculate start position for this sequence
        start_idx = idx * self.sequence_length
        
        # Extract exact sequence
        inputs = torch.tensor(
            self.token_ids[start_idx:start_idx + self.sequence_length], 
            dtype=torch.long
        )
        
        # Targets are shifted by 1 position
        # For the last sequence, we'll wrap around or pad with UNK
        if start_idx + self.sequence_length < len(self.token_ids):
            targets = torch.tensor(
                self.token_ids[start_idx + 1:start_idx + self.sequence_length + 1], 
                dtype=torch.long
            )
        else:
            # For the last sequence, use UNK for the final target
            target_tokens = self.token_ids[start_idx + 1:] + [0]  # 0 is UNK token
            targets = torch.tensor(target_tokens[:self.sequence_length], dtype=torch.long)
            
        return inputs, targets

def load_wikitext_data(subset='train', max_samples=None):
    """Load WikiText-103 data"""
    print(f"Loading WikiText-103 {subset} data...")
    
    dataset = datasets.load_dataset('wikitext', 'wikitext-103-raw-v1')
    texts = dataset[subset]['text']
    
    if max_samples:
        texts = texts[:max_samples]
    
    # Filter out empty lines - keep all text including short lines
    texts = [text for text in texts if text.strip()]
    
    print(f"Loaded {len(texts)} {subset} texts")
    return texts

def create_tokenized_data(texts, tokenizer):
    """Tokenize texts and concatenate into one long sequence - no EOS tokens"""
    print("Tokenizing texts...")
    
    all_tokens = []
    for i, text in enumerate(texts):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(texts)} texts")
            
        tokens = tokenizer.encode(text.strip())
        if tokens:  # Only add non-empty token sequences
            all_tokens.extend(tokens)
    
    print(f"Created continuous sequence with {len(all_tokens)} tokens")
    return all_tokens

def prepare_wikitext_datasets(vocab_size=10000, sequence_length=128, max_val_samples=500):
    """Prepare WikiText-103 datasets with word-level tokenization"""
    
    # Check if tokenizer already exists
    tokenizer_path = 'wikitext_tokenizer.pkl'
    
    if os.path.exists(tokenizer_path):
        print("Loading existing tokenizer...")
        tokenizer = SimpleWordTokenizer(vocab_size)
        tokenizer.load(tokenizer_path)
    else:
        print("Creating new tokenizer...")
        # Load all training data for vocabulary building
        train_texts = load_wikitext_data('train')
        
        # Build tokenizer
        tokenizer = SimpleWordTokenizer(vocab_size)
        tokenizer.build_vocab(train_texts, min_freq=2)
        
        # Save tokenizer
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    
    # Load and tokenize data
    train_texts = load_wikitext_data('train', max_val_samples)
    val_texts = load_wikitext_data('validation', max_val_samples)
    
    # Create token sequences
    train_tokens = create_tokenized_data(train_texts, tokenizer)
    val_tokens = create_tokenized_data(val_texts, tokenizer)
    
    # Create datasets
    train_dataset = WikiTextDataset(train_tokens, sequence_length)
    val_dataset = WikiTextDataset(val_tokens, sequence_length)
    
    print(f"\nDataset Summary:")
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset, tokenizer

class WikiTextDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for WikiText-103 with word-level tokenization"""
    
    def __init__(self, vocab_size=10000, sequence_length=128, batch_size=32, 
                 num_workers=4, max_val_samples=500):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_val_samples = max_val_samples
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.tokenizer = None
        
    def setup(self, stage=None):
        """Setup datasets"""
        if stage == "fit" or stage is None:
            print("Setting up WikiText-103 datasets...")
            self.train_dataset, self.val_dataset, self.tokenizer = prepare_wikitext_datasets(
                vocab_size=self.vocab_size,
                sequence_length=self.sequence_length,
                max_val_samples=self.max_val_samples
            )
            print(f"Vocabulary size: {len(self.tokenizer.word2idx)}")
            
    def train_dataloader(self):
        """Create training dataloader with transposition for CBR_RNN"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=lambda batch: self._transpose_batch(batch)
        )
    
    def val_dataloader(self):
        """Create validation dataloader with transposition for CBR_RNN"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=lambda batch: self._transpose_batch(batch)
        )
    
    def _transpose_batch(self, batch):
        """Simple function to transpose batch for CBR_RNN format"""
        inputs, targets = zip(*batch)
        inputs = torch.stack(inputs).transpose(0, 1)  # [batch, seq] -> [seq, batch]
        targets = torch.stack(targets).transpose(0, 1)  # [batch, seq] -> [seq, batch]
        return inputs, targets

# # Test the dataset and data module
# if __name__ == "__main__":
#     print("Testing WikiText DataModule...")
    
#     # Create test data module
#     data_module = WikiTextDataModule(
#         vocab_size=5000,
#         sequence_length=64,
#         batch_size=4,
#         max_val_samples=100
#     )
    
#     # Setup the data module
#     data_module.setup()
    
#     # Get dataloaders
#     train_loader = data_module.train_dataloader()
#     val_loader = data_module.val_dataloader()
    
#     print(f"\nTrain batches: {len(train_loader)}")
#     print(f"Val batches: {len(val_loader)}")
    
#     # Test a batch
#     batch = next(iter(train_loader))
#     inputs, targets = batch
    
#     print(f"\nBatch shapes (after transpose for CBR_RNN):")
#     print(f"Inputs: {inputs.shape}")  # Should be [seq_len, batch_size]
#     print(f"Targets: {targets.shape}")  # Should be [seq_len, batch_size]
    
#     # Show some example text
#     print(f"\nExample input text:")
#     print(data_module.tokenizer.decode(inputs[:, 0].tolist()))  # First sequence in batch
#     print(f"\nExample target text:")
#     print(data_module.tokenizer.decode(targets[:, 0].tolist()))
    
#     # Show vocabulary stats
#     print(f"\nVocabulary examples:")
#     for i in range(10):
#         print(f"{i}: '{data_module.tokenizer.idx2word[i]}'")
    
#     print("\n✅ DataModule test completed successfully!")