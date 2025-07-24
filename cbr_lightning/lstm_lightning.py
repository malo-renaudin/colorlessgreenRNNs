import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import math


class SimpleLSTM(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, 
                 num_layers=2, dropout=0.2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    param.data[(n//4):(n//2)].fill_(1)
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden and cell states"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def forward(self, input_ids, hidden=None):
        batch_size, seq_len = input_ids.size()
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, input_ids.device)
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to vocabulary size
        logits = self.output_projection(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden
    
    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        logits, _ = self(input_ids)
        
        # Shift logits and targets for next token prediction
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_targets = targets[..., 1:].contiguous()
        
        # Calculate loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_perplexity', perplexity, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        logits, _ = self(input_ids)
        
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_targets = targets[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
        
        perplexity = torch.exp(loss)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_perplexity', perplexity, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    