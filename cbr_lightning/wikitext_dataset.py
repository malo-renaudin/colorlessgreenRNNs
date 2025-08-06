import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import datasets
from collections import Counter, defaultdict
import pickle
import re
import json
import logging
import hashlib
import tempfile
import shutil
import fcntl
import os
import time
from typing import List, Dict, Tuple, Optional, Iterator, Generator
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import psutil
import gc


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration class for data processing"""
    vocab_size: int = 50000
    min_freq: int = 2
    sequence_length: int = 35
    add_special_tokens: bool = False
    overlap_sequences: bool = True
    max_memory_gb: float = 8.0
    batch_size_tokenization: int = 1000
    cache_chunk_size: int = 10000
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def get_hash(self) -> str:
        """Get configuration hash for cache validation"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class TokenizationStats:
    """Track tokenization statistics for debugging"""
    def __init__(self):
        self.total_texts = 0
        self.empty_texts = 0
        self.total_tokens = 0
        self.invalid_tokens = 0
        self.unk_tokens = 0
        self.sequences_created = 0
        
    def log_stats(self):
        logger.info(f"Tokenization Stats:")
        logger.info(f"  Total texts: {self.total_texts}")
        logger.info(f"  Empty texts: {self.empty_texts}")
        logger.info(f"  Total tokens: {self.total_tokens}")
        logger.info(f"  Invalid tokens (fixed): {self.invalid_tokens}")  
        logger.info(f"  UNK tokens: {self.unk_tokens}")
        logger.info(f"  Sequences created: {self.sequences_created}")


@contextmanager
def file_lock(filepath: Path):
    """Context manager for file locking to prevent race conditions"""
    lock_file = filepath.with_suffix(filepath.suffix + '.lock')
    try:
        with open(lock_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
    except IOError:
        logger.warning(f"Could not acquire lock for {filepath}, waiting...")
        time.sleep(1)
        with open(lock_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield
    finally:
        try:
            lock_file.unlink()
        except FileNotFoundError:
            pass


class MemoryMonitor:
    """Monitor memory usage and trigger cleanup when needed"""
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        
    def check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        memory_usage = psutil.Process().memory_info().rss
        return memory_usage < self.max_memory_bytes
    
    def cleanup_if_needed(self):
        """Force garbage collection if memory is high"""
        if not self.check_memory():
            logger.warning("High memory usage detected, running garbage collection")
            gc.collect()


class WordTokenizer:
    """Enhanced word-level tokenizer with better error handling and validation"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.BOS_TOKEN = '<bos>'
        self.EOS_TOKEN = '<eos>'
        
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self._compiled_patterns = self._compile_regex_patterns()
        
    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for better performance"""
        return {
            'section_headers': re.compile(r'=+ .+ =+'),
            'html_tags': re.compile(r'<[^>]+>'),
            'punctuation': re.compile(r'([.!?;,:])'),
            'whitespace': re.compile(r'\s+')
        }
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Optimized tokenization with compiled regex patterns"""
        if not text or not text.strip():
            return []
            
        try:
            text = text.lower().strip()
            
            # Apply pre-compiled patterns
            text = self._compiled_patterns['section_headers'].sub('', text)
            text = self._compiled_patterns['html_tags'].sub('', text)
            text = self._compiled_patterns['punctuation'].sub(r' \1 ', text)
            text = self._compiled_patterns['whitespace'].sub(' ', text)
            
            tokens = text.strip().split()
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary with batch processing and memory management"""
        logger.info("Building vocabulary...")
        memory_monitor = MemoryMonitor(self.config.max_memory_gb)
        
        # Process texts in batches to manage memory
        batch_size = self.config.batch_size_tokenization
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            logger.info(f"Processing vocab batch {batch_idx + 1}/{total_batches}")
            
            for text in batch_texts:
                if text and text.strip():
                    tokens = self._tokenize_text(text)
                    self.word_counts.update(tokens)
            
            # Memory cleanup if needed
            memory_monitor.cleanup_if_needed()
        
        logger.info(f"Total unique words before filtering: {len(self.word_counts)}")
        
        # Build vocabulary mappings
        self._build_vocab_mappings()
        
        logger.info(f"Final vocabulary size: {len(self.word2idx)}")
        logger.info("✓ Vocabulary building completed successfully")
    
    def _build_vocab_mappings(self):
        """Build word-to-index mappings with validation"""
        most_common = self.word_counts.most_common()
        
        self.word2idx = {}
        self.idx2word = {}
        
        # Add special tokens first
        current_id = 0
        for token in self.special_tokens:
            self.word2idx[token] = current_id
            self.idx2word[current_id] = token
            current_id += 1
        
        # Add vocabulary words
        max_words = self.config.vocab_size - len(self.special_tokens)
        words_added = 0
        
        for word, count in most_common:
            if (count >= self.config.min_freq and 
                current_id < self.config.vocab_size and 
                words_added < max_words):
                
                self.word2idx[word] = current_id
                self.idx2word[current_id] = word
                current_id += 1
                words_added += 1
        
        # Validate mappings
        assert len(self.word2idx) == len(self.idx2word), "Vocabulary mapping size mismatch"
        assert all(0 <= idx < len(self.word2idx) for idx in self.idx2word.keys()), "Invalid indices in vocabulary"
    
    def encode(self, text: str, stats: Optional[TokenizationStats] = None) -> List[int]:
        """Convert text to token IDs with statistics tracking"""
        tokens = self._tokenize_text(text)
        token_ids = []
        
        if stats:
            stats.total_tokens += len(tokens)
        
        for token in tokens:
            if token in self.word2idx:
                token_id = self.word2idx[token]
                # Validate token ID
                if 0 <= token_id < len(self.word2idx):
                    token_ids.append(token_id)
                else:
                    logger.warning(f"Invalid token ID {token_id} for token '{token}'")
                    token_ids.append(self.word2idx[self.UNK_TOKEN])
                    if stats:
                        stats.invalid_tokens += 1
            else:
                token_ids.append(self.word2idx[self.UNK_TOKEN])
                if stats:
                    stats.unk_tokens += 1
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text with validation"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.idx2word:
                tokens.append(self.idx2word[token_id])
            else:
                logger.warning(f"Unknown token ID: {token_id}")
                tokens.append(self.UNK_TOKEN)
        
        return ' '.join(tokens)
    
    def save(self, filepath: str) -> None:
        """Save tokenizer with atomic write operation"""
        tokenizer_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'config': self.config.to_dict(),
            'special_tokens': self.special_tokens,
            'version': '2.0'  # Version for compatibility checking
        }
        
        filepath = Path(filepath)
        
        try:
            with file_lock(filepath):
                # Use temporary file for atomic write
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, 
                                                 dir=filepath.parent, 
                                                 prefix=filepath.stem + '_tmp') as tmp_file:
                    pickle.dump(tokenizer_data, tmp_file)
                    tmp_filepath = tmp_file.name
                
                # Atomic move
                shutil.move(tmp_filepath, filepath)
                logger.info(f"Tokenizer saved to {filepath}")
                
        except Exception as e:
            logger.error(f"Error saving tokenizer: {e}")
            # Cleanup temp file if it exists
            if 'tmp_filepath' in locals() and Path(tmp_filepath).exists():
                Path(tmp_filepath).unlink()
            raise
    
    def load(self, filepath: str) -> None:
        """Load tokenizer with validation"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                tokenizer_data = pickle.load(f)
            
            # Version compatibility check
            if tokenizer_data.get('version', '1.0') != '2.0':
                logger.warning("Loading tokenizer from older version, some features may not work")
            
            self.word2idx = tokenizer_data['word2idx']
            self.idx2word = tokenizer_data['idx2word']
            self.special_tokens = tokenizer_data['special_tokens']
            
            # Validate loaded data
            self._validate_loaded_tokenizer()
            
            logger.info(f"Tokenizer loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def _validate_loaded_tokenizer(self):
        """Validate loaded tokenizer data"""
        assert len(self.word2idx) == len(self.idx2word), "Vocabulary mapping size mismatch"
        assert all(token in self.word2idx for token in self.special_tokens), "Missing special tokens"
        assert self.word2idx[self.PAD_TOKEN] == 0, "PAD token should have ID 0"
    
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


class StreamingDataset(IterableDataset):
    """Memory-efficient streaming dataset"""
    
    def __init__(self, sequences_generator: Generator, total_sequences: int):
        self.sequences_generator = sequences_generator
        self.total_sequences = total_sequences
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for sequence in self.sequences_generator:
            if len(sequence) < 2:
                continue
                
            # Input is all tokens except the last one
            input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
            # Target is all tokens except the first one
            target_ids = torch.tensor(sequence[1:], dtype=torch.long)
            
            yield input_ids, target_ids
    
    def __len__(self) -> int:
        return self.total_sequences


class CachedDataset(Dataset):
    """Dataset with chunked caching for memory efficiency"""
    
    def __init__(self, cache_dir: Path, config_hash: str, split_name: str):
        self.cache_dir = cache_dir
        self.config_hash = config_hash
        self.split_name = split_name
        self.chunk_files = list(self.cache_dir.glob(f"{split_name}_{config_hash}_chunk_*.pkl"))
        self.chunk_files.sort()
        
        # Load metadata
        metadata_file = self.cache_dir / f"{split_name}_{config_hash}_metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.total_sequences = self.metadata['total_sequences']
        self.sequences_per_chunk = self.metadata['sequences_per_chunk']
        
        # Cache for currently loaded chunk
        self._current_chunk_idx = -1
        self._current_chunk_data = None
        
        logger.info(f"Cached dataset loaded: {len(self.chunk_files)} chunks, {self.total_sequences} sequences")
    
    def __len__(self) -> int:
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_idx = idx // self.sequences_per_chunk
        local_idx = idx % self.sequences_per_chunk
        
        # Load chunk if not current
        if chunk_idx != self._current_chunk_idx:
            self._load_chunk(chunk_idx)
        
        sequence = self._current_chunk_data[local_idx]
        
        # Convert to tensors
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids
    
    def _load_chunk(self, chunk_idx: int):
        """Load a specific chunk into memory"""
        if chunk_idx >= len(self.chunk_files):
            raise IndexError(f"Chunk index {chunk_idx} out of range")
        
        chunk_file = self.chunk_files[chunk_idx]
        
        try:
            with open(chunk_file, 'rb') as f:
                self._current_chunk_data = pickle.load(f)
            self._current_chunk_idx = chunk_idx
            
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_idx}: {e}")
            raise


class SequenceProcessor:
    """Enhanced sequence processor with chunked caching and better error handling"""
    
    def __init__(self, tokenizer: WordTokenizer, config: ProcessingConfig, cache_dir: str = "sequence_cache"):
        self.tokenizer = tokenizer
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_monitor = MemoryMonitor(config.max_memory_gb)
    
    def _get_data_hash(self, texts: List[str]) -> str:
        """Generate hash of entire dataset for cache validation"""
        # Sample texts more thoroughly for better validation
        sample_size = min(100, len(texts))
        sample_indices = np.linspace(0, len(texts) - 1, sample_size, dtype=int)
        sample_texts = [texts[i] for i in sample_indices]
        
        combined_text = ''.join(sample_texts) + f"total_texts:{len(texts)}"
        return hashlib.sha256(combined_text.encode()).hexdigest()[:16]
    
    def _cache_exists(self, split_name: str, config_hash: str, data_hash: str) -> bool:
        """Check if valid cache exists"""
        metadata_file = self.cache_dir / f"{split_name}_{config_hash}_metadata.json"
        
        if not metadata_file.exists():
            return False
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Validate cache integrity
            return (metadata.get('data_hash') == data_hash and
                    metadata.get('config_hash') == config_hash and
                    self._validate_cache_files(split_name, config_hash, metadata))
        
        except Exception as e:
            logger.warning(f"Error validating cache: {e}")
            return False
    
    def _validate_cache_files(self, split_name: str, config_hash: str, metadata: dict) -> bool:
        """Validate that all cache chunk files exist"""
        expected_chunks = metadata.get('num_chunks', 0)
        
        for chunk_idx in range(expected_chunks):
            chunk_file = self.cache_dir / f"{split_name}_{config_hash}_chunk_{chunk_idx}.pkl"
            if not chunk_file.exists():
                logger.warning(f"Missing cache chunk: {chunk_file}")
                return False
        
        return True
    
    def _save_sequences_chunked(self, sequences: List[List[int]], split_name: str, 
                               config_hash: str, data_hash: str) -> None:
        """Save sequences in chunks to manage memory"""
        chunk_size = self.config.cache_chunk_size
        num_chunks = (len(sequences) + chunk_size - 1) // chunk_size
        
        logger.info(f"Saving {len(sequences)} sequences in {num_chunks} chunks...")
        
        try:
            # Save chunks
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(sequences))
                chunk_data = sequences[start_idx:end_idx]
                
                chunk_file = self.cache_dir / f"{split_name}_{config_hash}_chunk_{chunk_idx}.pkl"
                
                with file_lock(chunk_file):
                    with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                                     dir=self.cache_dir,
                                                     prefix=f"{split_name}_chunk_{chunk_idx}_tmp") as tmp_file:
                        pickle.dump(chunk_data, tmp_file)
                        tmp_filepath = tmp_file.name
                    
                    shutil.move(tmp_filepath, chunk_file)
                
                logger.info(f"Saved chunk {chunk_idx + 1}/{num_chunks}")
            
            # Save metadata
            metadata = {
                'split_name': split_name,
                'config_hash': config_hash,
                'data_hash': data_hash,
                'total_sequences': len(sequences),
                'sequences_per_chunk': chunk_size,
                'num_chunks': num_chunks,
                'config': self.config.to_dict(),
                'timestamp': time.time()
            }
            
            metadata_file = self.cache_dir / f"{split_name}_{config_hash}_metadata.json"
            with file_lock(metadata_file):
                with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                 dir=self.cache_dir,
                                                 prefix=f"{split_name}_metadata_tmp") as tmp_file:
                    json.dump(metadata, tmp_file, indent=2)
                    tmp_filepath = tmp_file.name
                
                shutil.move(tmp_filepath, metadata_file)
            
            logger.info(f"Cache saved successfully for {split_name}")
            
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            self._cleanup_partial_cache(split_name, config_hash)
            raise
    
    def _cleanup_partial_cache(self, split_name: str, config_hash: str):
        """Clean up partially written cache files"""
        pattern = f"{split_name}_{config_hash}_*"
        for cache_file in self.cache_dir.glob(pattern):
            try:
                cache_file.unlink()
                logger.info(f"Cleaned up partial cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Could not clean up {cache_file}: {e}")
    
    def process_texts(self, texts: List[str], split_name: str) -> Dataset:
        """Process texts into sequences with chunked caching"""
        config_hash = self.config.get_hash()
        data_hash = self._get_data_hash(texts)
        
        # Check if cache exists and is valid
        if self._cache_exists(split_name, config_hash, data_hash):
            logger.info(f"Loading cached sequences for {split_name}")
            return CachedDataset(self.cache_dir, config_hash, split_name)
        
        # Process sequences if not cached
        logger.info(f"Processing {len(texts)} texts for {split_name}...")
        sequences = self._create_sequences(texts, split_name)
        
        # Save to cache
        self._save_sequences_chunked(sequences, split_name, config_hash, data_hash)
        
        # Return cached dataset
        return CachedDataset(self.cache_dir, config_hash, split_name)
    
    def _create_sequences(self, texts: List[str], split_name: str) -> List[List[int]]:
        """Create sequences from texts with memory management"""
        sequences = []
        stats = TokenizationStats()
        vocab_size = len(self.tokenizer.word2idx)
        
        batch_size = self.config.batch_size_tokenization
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            if batch_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches} for {split_name}")
            
            batch_sequences = []
            
            for text in batch_texts:
                stats.total_texts += 1
                
                if not text or not text.strip():
                    stats.empty_texts += 1
                    continue
                
                try:
                    tokens = self.tokenizer.encode(text, stats)
                    
                    # Validate all tokens
                    valid_tokens = []
                    for token_id in tokens:
                        if 0 <= token_id < vocab_size:
                            valid_tokens.append(token_id)
                        else:
                            valid_tokens.append(self.tokenizer.unk_token_id)
                            stats.invalid_tokens += 1
                    
                    tokens = valid_tokens
                    
                    # Add special tokens if configured
                    if self.config.add_special_tokens:
                        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
                    
                    # Create sequences from tokens
                    text_sequences = self._create_sequences_from_tokens(tokens)
                    batch_sequences.extend(text_sequences)
                    stats.sequences_created += len(text_sequences)
                    
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
                    continue
            
            sequences.extend(batch_sequences)
            
            # Memory management
            self.memory_monitor.cleanup_if_needed()
        
        stats.log_stats()
        logger.info(f"Created {len(sequences)} total sequences for {split_name}")
        
        return sequences
    
    def _create_sequences_from_tokens(self, tokens: List[int]) -> List[List[int]]:
        """Create sequences from token list"""
        sequences = []
        
        if len(tokens) <= self.config.sequence_length:
            # If text is shorter than sequence length, pad or skip
            if len(tokens) > 1:  # Need at least 2 tokens for input/target
                padded = tokens + [self.tokenizer.pad_token_id] * (self.config.sequence_length + 1 - len(tokens))
                sequences.append(padded)
            return sequences
        
        if self.config.overlap_sequences:
            # Overlapping sequences (sliding window)
            for i in range(len(tokens) - self.config.sequence_length):
                sequence = tokens[i:i + self.config.sequence_length + 1]
                sequences.append(sequence)
        else:
            # Non-overlapping sequences
            for i in range(0, len(tokens) - self.config.sequence_length, self.config.sequence_length):
                if i + self.config.sequence_length + 1 <= len(tokens):
                    sequence = tokens[i:i + self.config.sequence_length + 1]
                    sequences.append(sequence)
        
        return sequences
    
    def clear_cache(self) -> None:
        """Clear all cached sequences"""
        try:
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file():
                    cache_file.unlink()
            logger.info(f"Cleared cache directory: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


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
    """Enhanced PyTorch Lightning DataModule with robust caching and error handling"""
    
    def __init__(
        self,
        data_path: str = "cbr_lightning/wikitext-103-local",
        tokenizer_path: Optional[str] = None,
        cache_dir: str = "sequence_cache",
        config: Optional[ProcessingConfig] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        force_reprocess: bool = False
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.cache_dir = cache_dir
        self.config = config or ProcessingConfig()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_reprocess = force_reprocess
        
        self.tokenizer = None
        self.sequence_processor = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self) -> None:
        """Validate local WikiText-103 dataset exists"""
        data_path = Path(self.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Local dataset not found at {self.data_path}")
        
        required_splits = ['train', 'validation', 'test']
        for split in required_splits:
            split_path = data_path / split
            if not split_path.exists():
                raise FileNotFoundError(f"Split '{split}' not found at {split_path}")
        
        logger.info(f"Using local WikiText-103 dataset from {self.data_path}")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets and tokenizer with enhanced error handling"""
        try:
            # Load dataset
            logger.info(f"Loading local dataset from {self.data_path}")
            wikitext = datasets.load_from_disk(self.data_path)
            
            # Initialize or load tokenizer
            self._setup_tokenizer(wikitext)
            
            # Initialize sequence processor
            self.sequence_processor = SequenceProcessor(
                tokenizer=self.tokenizer,
                config=self.config,
                cache_dir=self.cache_dir
            )
            
            # Clear cache if force reprocess is requested
            if self.force_reprocess:
                logger.info("Force reprocess requested, clearing cache...")
                self.sequence_processor.clear_cache()
            
            # Create datasets
            self._create_datasets(wikitext, stage)
            
        except Exception as e:
            logger.error(f"Error in setup: {e}")
            raise
    
    def _setup_tokenizer(self, wikitext):
        """Setup tokenizer with proper error handling"""
        if self.tokenizer_path and Path(self.tokenizer_path).exists():
            logger.info(f"Loading existing tokenizer from {self.tokenizer_path}")
            self.tokenizer = WordTokenizer(self.config)
            try:
                self.tokenizer.load(self.tokenizer_path)
                logger.info("Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                logger.info("Building new tokenizer...")
                self._build_new_tokenizer(wikitext)
        else:
            logger.info("Building new tokenizer...")
            self._build_new_tokenizer(wikitext)
    
    def _build_new_tokenizer(self, wikitext):
        """Build and save new tokenizer"""
        self.tokenizer = WordTokenizer(self.config)
        
        # Build vocabulary on training data
        train_texts = [text for text in wikitext['train']['text'] if text and text.strip()]
        logger.info(f"Building vocabulary from {len(train_texts)} training texts")
        
        self.tokenizer.build_vocab(train_texts)
        
        # Save tokenizer if path provided
        if self.tokenizer_path:
            try:
                self.tokenizer.save(self.tokenizer_path)
                logger.info(f"New tokenizer saved to {self.tokenizer_path}")
            except Exception as e:
                logger.error(f"Failed to save tokenizer: {e}")
    
    def _create_datasets(self, wikitext, stage: Optional[str]):
        """Create datasets for specified stage"""
        if stage == "fit" or stage is None:
            train_texts = [text for text in wikitext['train']['text'] if text and text.strip()]
            val_texts = [text for text in wikitext['validation']['text'] if text and text.strip()]
            
            logger.info(f"Creating training dataset from {len(train_texts)} texts")
            self.train_dataset = self.sequence_processor.process_texts(train_texts, "train")
            
            logger.info(f"Creating validation dataset from {len(val_texts)} texts")
            self.val_dataset = self.sequence_processor.process_texts(val_texts, "validation")
        
        if stage == "test" or stage is None:
            test_texts = [text for text in wikitext['test']['text'] if text and text.strip()]
            
            logger.info(f"Creating test dataset from {len(test_texts)} texts")
            self.test_dataset = self.sequence_processor.process_texts(test_texts, "test")
    
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn_cbr,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_cbr,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_cbr,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
    
    def clear_sequence_cache(self) -> None:
        """Clear all cached sequences"""
        if self.sequence_processor:
            self.sequence_processor.clear_cache()
        else:
            # Clear cache even if processor not initialized
            cache_dir = Path(self.cache_dir)
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*"):
                    if cache_file.is_file():
                        try:
                            cache_file.unlink()
                        except Exception as e:
                            logger.warning(f"Could not delete {cache_file}: {e}")
                logger.info(f"Cleared cache directory: {cache_dir}")
    
    def get_tokenizer(self) -> WordTokenizer:
        """Get the tokenizer instance"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call setup() first.")
        return self.tokenizer
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call setup() first.")
        return len(self.tokenizer.word2idx)





