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


class DualTimescaleSlotModel(pl.LightningModule):
    """
    Dual-timescale slot-based language model:
    - Fast timescale: Slot RNNs update every timestep (working memory)
    - Slow timescale: Main RNN updates every T timesteps (syntactic integration)
    - Prediction: Combines slot information + RNN context at every timestep
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 256,
        nslots: int = 8,
        slot_hidden_dim: int = 256,
        rnn_hidden_dim: int = 512,
        slot_type: str = "LSTM",  # "RNN", "LSTM", or "GRU" for slots
        rnn_type: str = "LSTM",   # "RNN", "LSTM", or "GRU" for main RNN
        T: int = 4,               # Slow timescale: RNN updates every T steps
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        max_seq_length: int = 128
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nslots = nslots
        self.slot_hidden_dim = slot_hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.slot_type = slot_type
        self.rnn_type = rnn_type
        self.T = T
        self.chunk_size = max_seq_length // T
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_seq_length = max_seq_length
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Slot RNNs (fast timescale) - one per slot
        self.slot_rnns = nn.ModuleList()
        for _ in range(nslots):
            if slot_type == "RNN":
                rnn_cell = nn.RNN(
                    input_size=rnn_hidden_dim + embedding_dim,
                    hidden_size=slot_hidden_dim,
                    batch_first=False,
                    dropout=dropout if slot_hidden_dim > 1 else 0
                )
            elif slot_type == "LSTM":
                rnn_cell = nn.LSTM(
                    input_size=rnn_hidden_dim + embedding_dim,
                    hidden_size=slot_hidden_dim,
                    batch_first=False,
                    dropout=dropout if slot_hidden_dim > 1 else 0
                )
            elif slot_type == "GRU":
                rnn_cell = nn.GRU(
                    input_size=rnn_hidden_dim + embedding_dim,
                    hidden_size=slot_hidden_dim,
                    batch_first=False,
                    dropout=dropout if slot_hidden_dim > 1 else 0
                )
            else:
                raise ValueError(f"Unsupported slot type: {slot_type}")
            
            self.slot_rnns.append(rnn_cell)
        
        # Main RNN (slow timescale)
        if rnn_type == "RNN":
            self.main_rnn = nn.RNN(
                input_size=nslots * slot_hidden_dim,
                hidden_size=rnn_hidden_dim,
                batch_first=False,
                dropout=dropout if rnn_hidden_dim > 1 else 0
            )
        elif rnn_type == "LSTM":
            self.main_rnn = nn.LSTM(
                input_size=nslots * slot_hidden_dim,
                hidden_size=rnn_hidden_dim,
                batch_first=False,
                dropout=dropout if rnn_hidden_dim > 1 else 0
            )
        elif rnn_type == "GRU":
            self.main_rnn = nn.GRU(
                input_size=nslots * slot_hidden_dim,
                hidden_size=rnn_hidden_dim,
                batch_first=False,
                dropout=dropout if rnn_hidden_dim > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Output projection: combines slot information + RNN context
        self.output_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(
            nslots * slot_hidden_dim + rnn_hidden_dim,
            vocab_size
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def init_hidden_states(self, batch_size: int, device: torch.device) -> Tuple:
        """Initialize hidden states for all RNNs"""
        
        # Initialize slot RNN hidden states
        slot_hidden_states = []
        for _ in range(self.nslots):
            if self.slot_type == "LSTM":
                h0 = torch.zeros(1, batch_size, self.slot_hidden_dim, device=device)
                c0 = torch.zeros(1, batch_size, self.slot_hidden_dim, device=device)
                slot_hidden_states.append((h0, c0))
            else:  # RNN or GRU
                h0 = torch.zeros(1, batch_size, self.slot_hidden_dim, device=device)
                slot_hidden_states.append(h0)
        
        # Initialize main RNN hidden state
        if self.rnn_type == "LSTM":
            main_h0 = torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device)
            main_c0 = torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device)
            main_hidden_state = (main_h0, main_c0)
        else:  # RNN or GRU
            main_hidden_state = torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device)
        
        return slot_hidden_states, main_hidden_state
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Dual-timescale forward pass
        Args:
            input_ids: [seq_length, batch_size] token ids
        Returns:
            logits: [seq_length, batch_size, vocab_size] next token predictions
        """
        seq_length, batch_size = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        embeddings = self.embedding(input_ids)  # [seq_length, batch_size, embedding_dim]
        embeddings = self.embedding_dropout(embeddings)
        
        # Initialize hidden states
        slot_hidden_states, main_hidden_state = self.init_hidden_states(batch_size, device)
        
        # Process sequence with dual timescales
        logits_sequence = []
        
        for t in range(seq_length):
            current_embedding = embeddings[t:t+1, :, :]  # [1, batch_size, embedding_dim]
            
            # Get current main RNN hidden state
            if self.rnn_type == "LSTM":
                current_main_hidden = main_hidden_state[0]  # [1, batch_size, rnn_hidden_dim]
            else:
                current_main_hidden = main_hidden_state  # [1, batch_size, rnn_hidden_dim]
            
            # FAST TIMESCALE: Update each slot RNN at every timestep
            current_slot_hiddens = []
            for slot_idx in range(self.nslots):
                # Slot input: concatenate main RNN hidden + current word embedding
                slot_input = torch.cat([current_main_hidden, current_embedding], dim=-1)
                
                # Update slot RNN
                slot_output, slot_hidden_states[slot_idx] = self.slot_rnns[slot_idx](
                    slot_input, slot_hidden_states[slot_idx]
                )
                
                # Extract hidden state for prediction
                if self.slot_type == "LSTM":
                    slot_hidden = slot_hidden_states[slot_idx][0]  # [1, batch_size, slot_hidden_dim]
                else:
                    slot_hidden = slot_hidden_states[slot_idx]  # [1, batch_size, slot_hidden_dim]
                
                current_slot_hiddens.append(slot_hidden)
            
            # SLOW TIMESCALE: Update main RNN every T timesteps
            if (t + 1) % self.T == 0 or t == seq_length - 1:
                # Concatenate all current slot hidden states
                slots_concat = torch.cat(current_slot_hiddens, dim=-1)  # [1, batch_size, nslots * slot_hidden_dim]
                
                # Update main RNN with slot information
                main_output, main_hidden_state = self.main_rnn(slots_concat, main_hidden_state)
                
                # Update current_main_hidden for next predictions
                if self.rnn_type == "LSTM":
                    current_main_hidden = main_hidden_state[0]
                else:
                    current_main_hidden = main_hidden_state
            
            # TOKEN PREDICTION: Combine slot information + main RNN context
            slots_concat = torch.cat(current_slot_hiddens, dim=-1)  # [1, batch_size, nslots * slot_hidden_dim]
            
            # Create prediction input: [slot_hiddens, main_rnn_hidden]
            prediction_input = torch.cat([
                slots_concat.squeeze(0),        # [batch_size, nslots * slot_hidden_dim]
                current_main_hidden.squeeze(0)  # [batch_size, rnn_hidden_dim]
            ], dim=-1)  # [batch_size, nslots * slot_hidden_dim + rnn_hidden_dim]
            
            # Apply dropout and project to vocabulary
            prediction_input = self.output_dropout(prediction_input)
            logits = self.output_projection(prediction_input)  # [batch_size, vocab_size]
            
            logits_sequence.append(logits.unsqueeze(0))  # [1, batch_size, vocab_size]
        
        # Concatenate all predictions
        output_logits = torch.cat(logits_sequence, dim=0)  # [seq_length, batch_size, vocab_size]
        
        return output_logits
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        inputs, targets = batch  # Both [seq_length, batch_size]
        
        # Forward pass
        logits = self(inputs)  # [seq_length, batch_size, vocab_size]
        
        # Calculate loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        
        # Calculate metrics
        perplexity = torch.exp(loss)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_perplexity', perplexity, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        inputs, targets = batch
        
        logits = self(inputs)
        
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        
        perplexity = torch.exp(loss)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).float().mean()
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_perplexity', perplexity, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
 


# def create_trainer(args):
#     """Create PyTorch Lightning trainer with callbacks"""
    
#     callbacks = []
    
#     # Model checkpoint
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=args.checkpoint_dir,
#         filename='dual-timescale-{epoch:02d}-{val_loss:.2f}',
#         monitor='val_loss',
#         mode='min',
#         save_top_k=3,
#         save_last=True,
#         verbose=True
#     )
#     callbacks.append(checkpoint_callback)
    
#     # Early stopping
#     if args.early_stopping_patience > 0:
#         early_stop_callback = EarlyStopping(
#             monitor='val_loss',
#             min_delta=0.001,
#             patience=args.early_stopping_patience,
#             verbose=True,
#             mode='min'
#         )
#         callbacks.append(early_stop_callback)
    
#     # Learning rate monitor
#     lr_monitor = LearningRateMonitor(logging_interval='epoch')
#     callbacks.append(lr_monitor)
    
#     # Logger
#     if args.use_wandb:
#         logger = WandbLogger(
#             project="dual-timescale-slots",
#             name=f"nslots-{args.nslots}-T-{args.T}",
#             save_dir=args.log_dir
#         )
#     else:
#         logger = TensorBoardLogger(
#             save_dir=args.log_dir,
#             name="dual-timescale-slots"
#         )
    
#     trainer = pl.Trainer(
#         max_epochs=args.max_epochs,
#         accelerator='auto',
#         devices='auto',
#         strategy='auto',
#         precision=args.precision,
#         gradient_clip_val=args.grad_clip,
#         accumulate_grad_batches=args.accumulate_grad_batches,
#         val_check_interval=args.val_check_interval,
#         check_val_every_n_epoch=args.check_val_every_n_epoch,
#         log_every_n_steps=args.log_every_n_steps,
#         callbacks=callbacks,
#         logger=logger,
#         enable_progress_bar=True,
#         enable_model_summary=True
#     )
    
#     return trainer


# def main():
#     parser = argparse.ArgumentParser(description='Train Dual-Timescale Slot Language Model')
    
#     # Model arguments
#     parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
#     parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
#     parser.add_argument('--nslots', type=int, default=8, help='Number of memory slots')
#     parser.add_argument('--slot_hidden_dim', type=int, default=256, help='Slot RNN hidden dimension')
#     parser.add_argument('--rnn_hidden_dim', type=int, default=512, help='Main RNN hidden dimension')
#     parser.add_argument('--slot_type', type=str, default='LSTM', choices=['RNN', 'LSTM', 'GRU'], help='Slot RNN type')
#     parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['RNN', 'LSTM', 'GRU'], help='Main RNN type')
#     parser.add_argument('--T', type=int, default=4, help='Slow timescale update frequency')
#     parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
#     # Training arguments
#     parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
#     parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
#     parser.add_argument('--sequence_length', type=int, default=128, help='Sequence length')
#     parser.add_argument('--max_epochs', type=int, default=20, help='Maximum epochs')
#     parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation')
#     parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
#     # Data arguments
#     parser.add_argument('--num_workers', type=int, default=4, help='Data loading workers')
#     parser.add_argument('--max_val_samples', type=int, default=1000, help='Max validation samples')
    
#     # Trainer arguments
#     parser.add_argument('--precision', type=str, default='32', help='Training precision')
#     parser.add_argument('--val_check_interval', type=float, default=1.0, help='Validation interval')
#     parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='Validation frequency')
#     parser.add_argument('--log_every_n_steps', type=int, default=50, help='Logging frequency')
#     parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    
#     # Paths
#     parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
#     parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
#     parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
#     parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume checkpoint')
    
#     args = parser.parse_args()
    
#     # Create directories
#     os.makedirs(args.checkpoint_dir, exist_ok=True)
#     os.makedirs(args.log_dir, exist_ok=True)
    
#     print("Setting up data module...")
#     # Import your WikiText data module
#     from wikitext_dataset import WikiTextDataModule
    
#     data_module = WikiTextDataModule(
#         vocab_size=args.vocab_size,
#         sequence_length=args.sequence_length,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         max_val_samples=args.max_val_samples
#     )
    
#     data_module.setup()
    
#     print("Creating model...")
#     model = DualTimescaleSlotModel(
#         vocab_size=len(data_module.tokenizer.word2idx),
#         embedding_dim=args.embedding_dim,
#         nslots=args.nslots,
#         slot_hidden_dim=args.slot_hidden_dim,
#         rnn_hidden_dim=args.rnn_hidden_dim,
#         slot_type=args.slot_type,
#         rnn_type=args.rnn_type,
#         T=args.T,
#         learning_rate=args.learning_rate,
#         weight_decay=args.weight_decay,
#         dropout=args.dropout,
#         max_seq_length=args.sequence_length
#     )
    
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
#     print(f"Vocabulary size: {len(data_module.tokenizer.word2idx)}")
#     print(f"Architecture: {args.nslots} {args.slot_type} slots, {args.rnn_type} main RNN, T={args.T}")
    
#     trainer = create_trainer(args)
    
#     print("Starting training...")
#     trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)
    
#     print("Training completed!")
#     print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


# if __name__ == "__main__":
#     main()