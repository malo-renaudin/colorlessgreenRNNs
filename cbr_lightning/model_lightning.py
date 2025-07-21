# model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from attention import MultiheadAttention

class CBR_RNN(pl.LightningModule):
    def __init__(self, ntoken, ninp, nhid, nheads, dropout=0.5, learning_rate=1e-3, 
                 temperature=1.0, gumbel_softmax=False, criterion='cross_entropy',
                 optimizer_type='adam', weight_decay=0.0, scheduler_type=None):
        super().__init__()
        
        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters()
        
        # Model architecture (your existing code)
        self.nheads = nheads
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.score_attn = nn.Softmax(dim=-1)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.q = nn.Linear(ninp + nhid, nhid)
        self.intermediate_h = nn.Linear(nhid * 4, nhid * 4)
        self.decoder = nn.Linear(nhid, ntoken)
        self.q_norm = torch.nn.LayerNorm(nhid)
        self.int_norm = torch.nn.LayerNorm(nhid * 4)
        self.f_norm = torch.nn.LayerNorm(nhid * 3)
        self.nhid = nhid
        self.final_h = nn.Linear(nhid * 4, nhid * 3)
        self.multihead_attn = MultiheadAttention(
            embed_dim=nhid, num_heads=nheads, batch_first=True, tau = temperature, gumbel=gumbel_softmax
        )
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.gumbel_softmax = gumbel_softmax
        self.criterion = criterion
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        
        self.init_weights()

    def init_weights(self):
        """Initialize model weights for better training dynamics"""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "norm" in name:
                    nn.init.ones_(param)
                elif "encoder" in name:
                    nn.init.normal_(param, mean=0, std=0.01)
                elif "decoder" in name:
                    nn.init.normal_(param, mean=0, std=0.01)
                else:
                    nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="tanh")
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_cache(self, observation, nheads):
        """Initialize hidden state and attention caches"""
        if len(observation.size()) > 1:
            bsz = observation.size(dim=-1)
        else:
            bsz = 1

        hidden = torch.zeros(1, bsz, self.nhid).to(self.device) 
        if nheads == 1:
            key_cache = torch.zeros(bsz, 1, 1, self.nhid).to(self.device) 
            value_cache = torch.zeros(bsz, 1, 1, self.nhid).to(self.device) 
        elif nheads > 1:
            key_cache = torch.zeros(bsz, 1, self.nhid).to(self.device) 
            value_cache = torch.zeros(bsz, 1, self.nhid).to(self.device) 
        return hidden, key_cache, value_cache

    def update_cache(self, key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i, nheads):
        hidden_i = hidden_i.unsqueeze(0)
        hidden = torch.cat((hidden, hidden_i), dim=0)
        if nheads == 1:
            key_cache_i = key_cache_i.unsqueeze(1).unsqueeze(1)
            value_cache_i = value_cache_i.unsqueeze(1).unsqueeze(1)
            key_cache = torch.cat((key_cache, key_cache_i), dim=2)
            value_cache = torch.cat((value_cache, value_cache_i), dim=2)
        else:
            key_cache_i = key_cache_i.unsqueeze(1)
            value_cache_i = value_cache_i.unsqueeze(1)
            key_cache = torch.cat((key_cache, key_cache_i), dim=1)
            value_cache = torch.cat((value_cache, value_cache_i), dim=1)
            
        return key_cache, value_cache, hidden

    

    def intermediate_layers(self, i, emb, query, attn, hidden):
        intermediate_input = torch.cat((emb[i], query, attn, hidden[-1]), -1)
        intermediate = self.drop(
            self.tanh(self.int_norm(self.intermediate_h(intermediate_input)))
        )
        final_output = self.drop(self.tanh(self.f_norm(self.final_h(intermediate))))
        key_cache_i, value_cache_i, hidden_i = final_output.split(self.nhid, dim=-1)
        return key_cache_i, value_cache_i, hidden_i

    def get_query(self, emb, hidden):
        combined = torch.cat((emb, hidden[-1]), -1)
        query = self.drop(self.tanh(self.q_norm(self.q(combined))))
        query = query.unsqueeze(1)
        return query

    def forward(self, observation, initial_cache=None, nheads=None, temperature=None, gumbel_softmax=None):
        # Use instance variables if not provided
        nheads = nheads if nheads is not None else self.nheads
        temperature = temperature if temperature is not None else self.temperature
        gumbel_softmax = gumbel_softmax if gumbel_softmax is not None else self.gumbel_softmax
        
        seq_len = observation.size(0)
        
        if initial_cache is None:
            hidden, key_cache, value_cache = self.init_cache(observation, nheads)
        else:
            hidden, key_cache, value_cache = initial_cache
            
        emb = self.drop(self.encoder(observation))
        
        for i in range(seq_len):
            query = self.get_query(emb[i], hidden)
            attn, query = self.multihead_attn(query, key_cache, value_cache, nheads, temperature, gumbel_softmax)
            key_cache_i, value_cache_i, hidden_i = self.intermediate_layers(i, emb, query, attn, hidden)
            key_cache, value_cache, hidden = self.update_cache(key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i, nheads)

        decoded = self.decoder(hidden[1:])
        return decoded, hidden

    def training_step(self, batch, batch_idx):
        # Assumes batch is (input_seq, target_seq) or modify based on your data format
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            input_seq, target = batch
        else:
            # If your batch format is different, adjust accordingly
            input_seq = batch[:-1]  # All but last token as input
            target = batch[1:]      # All but first token as target
        
        output, _ = self(input_seq)
        
        # Reshape for loss computation
        if self.criterion == 'cross_entropy':
            output = output.view(-1, output.size(-1))
            target = target.view(-1)
            loss = F.cross_entropy(output, target)
        elif self.criterion == 'mse':
            loss = F.mse_loss(output, target)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            input_seq, target = batch
        else:
            input_seq = batch[:-1]
            target = batch[1:]
        
        output, _ = self(input_seq)
        
        if self.criterion == 'cross_entropy':
            output = output.view(-1, output.size(-1))
            target = target.view(-1)
            loss = F.cross_entropy(output, target)
            
            # Calculate accuracy for classification
            pred = torch.argmax(output, dim=-1)
            acc = (pred == target).float().mean()
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
            
        elif self.criterion == 'mse':
            loss = F.mse_loss(output, target)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        
        if self.scheduler_type is None:
            return optimizer
        elif self.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            return [optimizer], [scheduler]
        elif self.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            return [optimizer], [scheduler]
        elif self.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'
            }
        else:
            return optimizer