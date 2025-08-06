#!/usr/bin/env python3
"""
Complete Training Script for CBR_RNN on WikiText-103
with Enhanced Data Processing and Robust Error Handling
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import argparse
import os
import time
import sys
import logging
from pathlib import Path
import json
from typing import List, Dict, Any
import psutil

# Import your modules
from model_lightning import CBR_RNN
from wikitext_dataset import (
    WikiTextDataModule, 
    ProcessingConfig,
    WordTokenizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration class for training parameters"""
    def __init__(self, args):
        # Store all arguments
        self.args = args
        
        # Create processing config
        self.processing_config = ProcessingConfig(
            vocab_size=args.vocab_size,
            min_freq=args.min_freq,
            sequence_length=args.sequence_length,
            add_special_tokens=args.add_special_tokens,
            overlap_sequences=args.overlap_sequences,
            max_memory_gb=args.max_memory_gb,
            batch_size_tokenization=args.batch_size_tokenization,
            cache_chunk_size=args.cache_chunk_size
        )
    
    def save(self, filepath: str):
        """Save training configuration"""
        config_dict = {
            'args': vars(self.args),
            'processing_config': self.processing_config.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Training configuration saved to {filepath}")


def create_callbacks(args) -> List[pl.Callback]:
    """Create PyTorch Lightning callbacks with enhanced configuration"""
    callbacks = []
    
    # Model checkpointing with better monitoring
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, args.experiment_name),
        filename='{epoch:02d}-{step}-{val_loss:.4f}-{val_ppl:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=args.save_top_k,
        save_last=True,
        every_n_epochs=args.checkpoint_every_n_epochs,
        save_weights_only=False,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping with better configuration
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=args.early_stopping_delta,
            patience=args.patience,
            mode='min',
            verbose=True,
            strict=True,
            check_finite=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(
        logging_interval='step',
        log_momentum=True
    )
    callbacks.append(lr_monitor)
    
    return callbacks


def create_loggers(args) -> List[pl.loggers.Logger]:
    """Create loggers with enhanced configuration"""
    loggers = []
    
    # Create version string with key hyperparameters
    version_str = (f"v{args.vocab_size}_{args.sequence_length}_{args.batch_size}_"
                  f"lr{args.learning_rate}_nh{args.nhid}_ni{args.ninp}")
    
    # TensorBoard logger with better organization
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
        version=version_str,
        log_graph=True,
        default_hp_metric=False
    )
    loggers.append(tb_logger)
    
    # CSV logger for easy analysis
    csv_logger = CSVLogger(
        save_dir=args.log_dir,
        name=f"{args.experiment_name}_csv",
        version=version_str
    )
    loggers.append(csv_logger)
    
    return loggers


def create_trainer(args) -> pl.Trainer:
    """Create PyTorch Lightning trainer with enhanced configuration"""
    
    callbacks = create_callbacks(args)
    loggers = create_loggers(args)
    
    # Configure trainer with better defaults
    trainer_kwargs = {
        'max_epochs': args.epochs,
        'accelerator': 'auto',
        'devices': args.devices,
        'precision': args.precision,
        'gradient_clip_val': args.clip_grad,
        'gradient_clip_algorithm': 'norm',
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'log_every_n_steps': args.log_every_n_steps,
        'val_check_interval': args.val_check_interval,
        'callbacks': callbacks,
        'logger': loggers,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'deterministic': args.deterministic,
        'limit_train_batches': args.limit_train_batches,
        'limit_val_batches': args.limit_val_batches,
        'fast_dev_run': args.fast_dev_run,
        'enable_checkpointing': not args.no_checkpointing,
        'detect_anomaly': args.detect_anomaly,
        'profiler': args.profiler if args.profiler != 'none' else None,
        'sync_batchnorm': args.devices > 1,
    }
    
    # Add strategy for multi-GPU training
    if args.devices > 1:
        trainer_kwargs['strategy'] = 'ddp_find_unused_parameters_true'
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    return trainer


def print_system_info():
    """Print system information for debugging"""
    logger.info("="*80)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*80)
    
    # Python and PyTorch info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"PyTorch Lightning version: {pl.__version__}")
    
    # Hardware info
    logger.info(f"CPU count: {os.cpu_count()}")
    memory_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"Total memory: {memory_gb:.1f} GB")
    
    # GPU info
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_props.name} ({gpu_props.total_memory // 1024**2} MB)")
    else:
        logger.info("CUDA available: False")
    
    logger.info("="*80)


def print_model_info(model, data_module):
    """Print detailed model and data information"""
    logger.info("="*80)
    logger.info("MODEL & DATA CONFIGURATION")
    logger.info("="*80)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model Architecture:")
    logger.info(f"  Total Parameters: {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")
    logger.info(f"  Model Size (MB): {total_params * 4 / 1024**2:.2f}")
    logger.info(f"  Embedding Dim: {model.hparams.ninp}")
    logger.info(f"  Hidden Dim: {model.hparams.nhid}")
    logger.info(f"  Attention Heads: {model.hparams.nheads}")
    logger.info(f"  Vocabulary Size: {model.hparams.ntoken}")
    logger.info(f"  Dropout: {model.hparams.dropout}")
    
    # Attention configuration
    logger.info(f"\nAttention Configuration:")
    logger.info(f"  Gumbel Softmax: {model.gumbel_softmax}")
    logger.info(f"  Temperature: {model.temperature}")
    logger.info(f"  Compressed Dim: {getattr(model.hparams, 'compressed_dim', 'N/A')}")
    
    # Data info
    logger.info(f"\nDataset Information:")
    logger.info(f"  Vocab Size: {data_module.get_vocab_size()}")
    logger.info(f"  Sequence Length: {data_module.config.sequence_length}")
    logger.info(f"  Batch Size: {data_module.batch_size}")
    logger.info(f"  Overlap Sequences: {data_module.config.overlap_sequences}")
    logger.info(f"  Add Special Tokens: {data_module.config.add_special_tokens}")
    
    # Dataset sizes
    try:
        logger.info(f"  Train Dataset: {len(data_module.train_dataset):,} sequences")
        logger.info(f"  Val Dataset: {len(data_module.val_dataset):,} sequences")
        if data_module.test_dataset:
            logger.info(f"  Test Dataset: {len(data_module.test_dataset):,} sequences")
    except Exception as e:
        logger.warning(f"Could not get dataset sizes: {e}")
    
    logger.info("="*80)


def validate_configuration(args):
    """Validate training configuration"""
    logger.info("Validating configuration...")
    
    # Check data paths
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")
    
    # Check memory limits
    available_memory_gb = psutil.virtual_memory().total / (1024**3)
    if args.max_memory_gb > available_memory_gb * 0.9:
        logger.warning(f"Max memory ({args.max_memory_gb}GB) is close to available memory ({available_memory_gb:.1f}GB)")
    
    # Check batch size vs sequence length
    estimated_memory_per_batch = (args.batch_size * args.sequence_length * 4) / (1024**2)  # MB
    if estimated_memory_per_batch > 1000:  # > 1GB per batch
        logger.warning(f"Large memory usage per batch estimated: {estimated_memory_per_batch:.1f}MB")
    
    # Check GPU memory if using CUDA
    if torch.cuda.is_available() and args.devices > 0:
        gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        if estimated_memory_per_batch * 10 > gpu_memory_mb:  # rough estimate
            logger.warning(f"Batch size might be too large for GPU memory")
    
    logger.info("✓ Configuration validation passed")


def run_quick_validation(model, data_module, args):
    """Run quick validation before training"""
    if args.fast_dev_run:
        return
    
    logger.info("Running quick validation check...")
    
    try:
        # Test model creation and forward pass
        model.eval()
        
        # Get a validation batch
        val_loader = data_module.val_dataloader()
        test_batch = next(iter(val_loader))
        inputs, targets = test_batch
        
        logger.info(f"Batch shapes - Input: {inputs.shape}, Target: {targets.shape}")
        
        # Test forward pass
        with torch.no_grad():
            output,hidden = model(inputs)
            logger.info(f"Model output shape: {output.shape}")
            
            # Test validation step
            val_loss = model.validation_step(test_batch, 0)
            logger.info(f"✅ Validation check passed - loss: {val_loss:.4f}")
        
        # Test training step
        model.train()
        train_loader = data_module.train_dataloader()
        train_batch = next(iter(train_loader))
        train_loss = model.training_step(train_batch, 0)
        logger.info(f"✅ Training step check passed - loss: {train_loss:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Validation check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train CBR_RNN on WikiText-103 with Enhanced Data Processing')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='cbr_lightning/wikitext-103-local',
                       help='Path to local WikiText-103 dataset')
    parser.add_argument('--vocab_size', type=int, default=50000, 
                       help='Vocabulary size')
    parser.add_argument('--min_freq', type=int, default=2,
                       help='Minimum frequency for vocabulary')
    parser.add_argument('--sequence_length', type=int, default=35, 
                       help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--add_special_tokens', action='store_true',
                       help='Add BOS/EOS tokens to sequences')
    parser.add_argument('--overlap_sequences', action='store_true', default=True,
                       help='Use overlapping sequences (sliding window)')
    
    # Enhanced data processing parameters
    parser.add_argument('--max_memory_gb', type=float, default=8.0,
                       help='Maximum memory usage in GB')
    parser.add_argument('--batch_size_tokenization', type=int, default=1000,
                       help='Batch size for tokenization')
    parser.add_argument('--cache_chunk_size', type=int, default=10000,
                       help='Size of cache chunks')
    parser.add_argument('--cache_dir', type=str, default='cbr_lightning/sequence_cache',
                       help='Directory for sequence cache')
    parser.add_argument('--tokenizer_path', type=str, default='cbr_lightning/tokenizer.pkl',
                       help='Path to save/load tokenizer')
    parser.add_argument('--force_reprocess', action='store_true',
                       help='Force reprocessing of data (ignore cache)')
    
    # Model architecture parameters
    parser.add_argument('--ninp', type=int, default=512, 
                       help='Embedding dimension')
    parser.add_argument('--nhid', type=int, default=1024, 
                       help='Hidden dimension')
    parser.add_argument('--nheads', type=int, default=1, 
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, 
                       help='Dropout rate')
    parser.add_argument('--compressed_dim', type=int, default=1, 
                       help='Size of the sequence length dimension in the compressed cache')
    
    # Gumbel Softmax parameters
    parser.add_argument('--gumbel_softmax', action='store_true', 
                       help='Use Gumbel Softmax instead of regular softmax')
    parser.add_argument('--temperature', type=float, default=1.0, 
                       help='Temperature for Gumbel Softmax')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='Weight decay')
    parser.add_argument('--optimizer_type', type=str, default='adamw', 
                       choices=['adamw', 'sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--scheduler_type', type=str, default='plateau', 
                       choices=['cosine', 'step', 'plateau', 'none'], help='Learning rate scheduler')
    
    # Training setup
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of epochs')
    parser.add_argument('--devices', type=int, default=1, 
                       help='Number of devices to use')
    parser.add_argument('--precision', type=str, default='16-mixed', 
                       choices=['16-mixed', '32', 'bf16-mixed'], help='Training precision')
    parser.add_argument('--clip_grad', type=float, default=1.0, 
                       help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, 
                       help='Accumulate gradients over N batches')
    
    # Logging and checkpointing
    parser.add_argument('--experiment_name', type=str, default='cbr_rnn', 
                       help='Experiment name')
    parser.add_argument('--log_dir', type=str, default='lightning_logs', 
                       help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                       help='Directory for checkpoints')
    parser.add_argument('--log_every_n_steps', type=int, default=100, 
                       help='Log every N steps')
    parser.add_argument('--val_check_interval', type=float, default=1.0, 
                       help='Validation check interval (epochs)')
    parser.add_argument('--save_top_k', type=int, default=3,
                       help='Save top k checkpoints')
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=1,
                       help='Checkpoint every N epochs')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', 
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=7, 
                       help='Early stopping patience')
    parser.add_argument('--early_stopping_delta', type=float, default=0.001,
                       help='Minimum change to qualify as improvement')
    
    # Testing/debugging
    parser.add_argument('--limit_train_batches', type=float, default=1000000000000, 
                       help='Limit training batches (for testing)')
    parser.add_argument('--limit_val_batches', type=float, default=100000000000, 
                       help='Limit validation batches (for testing)')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of dataloader workers')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run 1 batch of train/val/test for debugging')
    parser.add_argument('--deterministic', action='store_true',
                       help='Enable deterministic training for reproducibility')
    parser.add_argument('--no_checkpointing', action='store_true',
                       help='Disable checkpointing (for debugging)')
    parser.add_argument('--detect_anomaly', action='store_true',
                       help='Enable anomaly detection for debugging')
    parser.add_argument('--profiler', type=str, default='none',
                       choices=['none', 'simple', 'advanced', 'pytorch'],
                       help='Profiler type')
    
    # Resume training
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Print system info
    print_system_info()
    
    # Validate configuration
    validate_configuration(args)
    
    # Create training configuration
    training_config = TrainingConfig(args)
    
    # Print configuration
    logger.info("="*80)
    logger.info("CBR_RNN Training on WikiText-103 (Enhanced)")
    logger.info("="*80)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Data: vocab_size={args.vocab_size}, seq_len={args.sequence_length}, batch_size={args.batch_size}")
    logger.info(f"Model: ninp={args.ninp}, nhid={args.nhid}, nheads={args.nheads}")
    logger.info(f"Attention: gumbel={args.gumbel_softmax}, temp={args.temperature}")
    logger.info(f"Training: lr={args.learning_rate}, epochs={args.epochs}, devices={args.devices}")
    logger.info(f"Memory: max_memory={args.max_memory_gb}GB, cache_chunks={args.cache_chunk_size}")
    logger.info("="*80)
    
    # Set random seed for reproducibility
    if args.deterministic:
        pl.seed_everything(42, workers=True)
        logger.info("✓ Deterministic training enabled")
    
    # Create directories
    for directory in [args.log_dir, args.checkpoint_dir, args.cache_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.checkpoint_dir, args.experiment_name, 'config.json')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    training_config.save(config_path)
    
    # Set up data module with enhanced processing
    logger.info("Setting up enhanced data module...")
    try:
        data_module = WikiTextDataModule(
            data_path=args.data_path,
            tokenizer_path=args.tokenizer_path,
            cache_dir=args.cache_dir,
            config=training_config.processing_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force_reprocess=args.force_reprocess
        )
        
        # Prepare and setup data
        data_module.prepare_data()
        data_module.setup("fit")
        
        logger.info(f"✓ Data module setup completed")
        logger.info(f"Vocabulary size: {data_module.get_vocab_size()}")
        
        # Print data info
        if hasattr(data_module, 'train_dataset') and data_module.train_dataset:
            logger.info(f"Training samples: {len(data_module.train_dataset):,}")
        if hasattr(data_module, 'val_dataset') and data_module.val_dataset:
            logger.info(f"Validation samples: {len(data_module.val_dataset):,}")
            
    except Exception as e:
        logger.error(f"Failed to setup data module: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Debug batch data
    logger.info("Debugging batch data...")
    try:
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        inputs, targets = batch
        
        vocab_size = data_module.get_vocab_size()
        logger.info(f"Input shape: {inputs.shape}")  # Should be (seq_len, batch_size)
        logger.info(f"Target shape: {targets.shape}")  # Should be (seq_len, batch_size)
        logger.info(f"Input min/max: {inputs.min().item()}/{inputs.max().item()}")
        logger.info(f"Target min/max: {targets.min().item()}/{targets.max().item()}")
        logger.info(f"Vocabulary size: {vocab_size}")
        
        # Validate all token IDs are within bounds
        if inputs.max().item() >= vocab_size or inputs.min().item() < 0:
            raise ValueError(f"Input tokens out of bounds: min={inputs.min().item()}, max={inputs.max().item()}, vocab_size={vocab_size}")
        if targets.max().item() >= vocab_size or targets.min().item() < 0:
            raise ValueError(f"Target tokens out of bounds: min={targets.min().item()}, max={targets.max().item()}, vocab_size={vocab_size}")
        
        logger.info("✓ All token IDs are within vocabulary bounds")
        
    except Exception as e:
        logger.error(f"Error debugging batch data: {e}")
        sys.exit(1)
    
    # Create model
    logger.info("Creating model...")
    try:
        model = CBR_RNN(
            ntoken=data_module.get_vocab_size(),
            ninp=args.ninp,
            nhid=args.nhid,
            nheads=args.nheads,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            gumbel_softmax=args.gumbel_softmax,
            criterion='cross_entropy',
            optimizer_type=args.optimizer_type,
            weight_decay=args.weight_decay,
            scheduler_type=args.scheduler_type,
            seq_len=args.sequence_length,
            compressed_dim=args.compressed_dim,
        )
        
        logger.info("✓ Model created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print model and data info
    print_model_info(model, data_module)
    
    # Run quick validation
    run_quick_validation(model, data_module, args)
    
    # Create trainer
    logger.info("Creating trainer...")
    try:
        trainer = create_trainer(args)
        logger.info("✓ Trainer created successfully")
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        sys.exit(1)
    
    # Train model
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    start_time = time.time()
    
    try:
        if args.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)
        else:
            trainer.fit(model, data_module)
        
        training_time = time.time() - start_time
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETED")
        logger.info("="*80)
        logger.info(f"Training time: {training_time/60:.2f} minutes")
        
        # Print final metrics
        if hasattr(trainer, 'callback_metrics'):
            for metric, value in trainer.callback_metrics.items():
                if 'val_' in metric:
                    logger.info(f"{metric}: {value}")
        
        # Save final model
        if not args.no_checkpointing:
            final_model_path = os.path.join(args.checkpoint_dir, args.experiment_name, 'final_model.ckpt')
            trainer.save_checkpoint(final_model_path)
            logger.info(f"Final model saved to: {final_model_path}")
        
        logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.info("="*50)
        logger.info("Training interrupted by user")
        logger.info("="*50)
        
    except Exception as e:
        logger.error("="*50)
        logger.error(f"Training failed with error: {e}")
        logger.error("="*50)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
