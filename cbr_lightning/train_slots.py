import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Tuple
import os
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from slots import DualTimescaleSlotModel

def create_trainer(args):
    """Create PyTorch Lightning trainer with callbacks"""
    
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='dual-timescale-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if args.early_stopping_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=args.early_stopping_patience,
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Logger
    if args.use_wandb:
        logger = WandbLogger(
            project="dual-timescale-slots",
            name=f"nslots-{args.nslots}-T-{args.T}",
            save_dir=args.log_dir
        )
    else:
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name="dual-timescale-slots"
        )
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices='auto',
        strategy='auto',
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description='Train Dual-Timescale Slot Language Model')
    
    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--nslots', type=int, default=8, help='Number of memory slots')
    parser.add_argument('--slot_hidden_dim', type=int, default=256, help='Slot RNN hidden dimension')
    parser.add_argument('--rnn_hidden_dim', type=int, default=512, help='Main RNN hidden dimension')
    parser.add_argument('--slot_type', type=str, default='LSTM', choices=['RNN', 'LSTM', 'GRU'], help='Slot RNN type')
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['RNN', 'LSTM', 'GRU'], help='Main RNN type')
    parser.add_argument('--T', type=int, default=8, help='Slow timescale update frequency')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=64, help='Sequence length')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Data arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Data loading workers')
    parser.add_argument('--max_val_samples', type=int, default=1000, help='Max validation samples')
    
    # Trainer arguments
    parser.add_argument('--precision', type=str, default='32', help='Training precision')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='Validation interval')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='Validation frequency')
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='Logging frequency')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume checkpoint')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("Setting up data module...")
    # Import your WikiText data module
    from wikitext_dataset import WikiTextDataModule
    
    data_module = WikiTextDataModule(
        vocab_size=args.vocab_size,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_val_samples=args.max_val_samples
    )
    
    data_module.setup()
    
    print("Creating model...")
    model = DualTimescaleSlotModel(
        vocab_size=len(data_module.tokenizer.word2idx),
        embedding_dim=args.embedding_dim,
        nslots=args.nslots,
        slot_hidden_dim=args.slot_hidden_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        slot_type=args.slot_type,
        rnn_type=args.rnn_type,
        T=args.T,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        max_seq_length=args.sequence_length
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Vocabulary size: {len(data_module.tokenizer.word2idx)}")
    print(f"Architecture: {args.nslots} {args.slot_type} slots, {args.rnn_type} main RNN, T={args.T}")
    
    trainer = create_trainer(args)
    
    print("Starting training...")
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)
    
    print("Training completed!")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()