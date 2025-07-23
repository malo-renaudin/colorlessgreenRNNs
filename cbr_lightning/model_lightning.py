# model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from attention import MultiheadAttention

class CBR_RNN(pl.LightningModule):
    def __init__(self, ntoken, ninp, nhid, nheads, seq_len, compressed_dim, dropout=0.5, learning_rate=1e-3, 
                 temperature=1.0, gumbel_softmax=False, criterion='cross_entropy',
                 optimizer_type='adam', weight_decay=0.0, scheduler_type=None):
        super().__init__()
        
        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters()
        self.epoch_cache = None
        self.seq_len = seq_len
        self.compressed_dim = compressed_dim
        self.nheads = nheads
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.score_attn = nn.Softmax(dim=-1)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.q = nn.Linear(ninp + nhid, nhid)
        self.intermediate_h = nn.Linear(nhid * 3 + ninp, nhid * 4)
        self.decoder = nn.Linear(nhid, ntoken)
        self.q_norm = torch.nn.LayerNorm(nhid)
        self.int_norm = torch.nn.LayerNorm(nhid * 4)
        self.f_norm = torch.nn.LayerNorm(nhid * 3)
        self.nhid = nhid
        self.final_h = nn.Linear(nhid * 4, nhid * 3)
        self.multihead_attn = MultiheadAttention(
            embed_dim=nhid, num_heads=nheads, batch_first=True
        )
        self.hidden_compress = nn.Linear(nhid*(seq_len+compressed_dim), nhid*compressed_dim)
        self.key_compress = nn.Linear(nhid*(seq_len+compressed_dim), nhid*compressed_dim) 
        self.value_compress = nn.Linear(nhid*(seq_len+compressed_dim), nhid*compressed_dim)
        self.hidden_compress_norm = nn.LayerNorm(nhid * compressed_dim)
        self.key_compress_norm = nn.LayerNorm(nhid * compressed_dim)
        self.value_compress_norm = nn.LayerNorm(nhid * compressed_dim)
        
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
                elif "compress" in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="tanh")
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_cache(self, observation):
        """Initialize hidden state and attention caches"""
        if len(observation.size()) > 1:
            bsz = observation.size(dim=-1)
        else:
            bsz = 1

        hidden = torch.zeros(self.compressed_dim, bsz, self.nhid).to(self.device) 
        key_cache = torch.zeros(bsz, self.compressed_dim, self.nhid).to(self.device) 
        value_cache = torch.zeros(bsz, self.compressed_dim, self.nhid).to(self.device) 
        return hidden, key_cache, value_cache

    def update_cache(self, key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i):
        hidden_i = hidden_i.unsqueeze(0)
        hidden = torch.cat((hidden, hidden_i), dim=0)
        key_cache_i = key_cache_i.unsqueeze(1)
        value_cache_i = value_cache_i.unsqueeze(1)
        key_cache = torch.cat((key_cache, key_cache_i), dim=1)
        value_cache = torch.cat((value_cache, value_cache_i), dim=1)
            
        return key_cache, value_cache, hidden

    def compress_cache(self, hidden, key_cache, value_cache):
        """
        Learned projection from [bsz, seq_len, nhid] to [bsz, compressed_dim, nhid]
        """
        # For hidden: [seq_len, batch, nhid] -> [batch, seq_len, nhid] -> [batch, compressed_dim, nhid] -> [compressed_dim, batch, nhid]
        hidden_reshaped = hidden.transpose(0, 1)  # [batch, seq_len, nhid]
        batch_size, seq_len, nhid = hidden_reshaped.shape

        hidden_flat = hidden_reshaped .reshape(batch_size, -1) 
        hidden_proj = self.drop(self.tanh(self.hidden_compress_norm(self.hidden_compress(hidden_flat))))  # [batch, nhid * compressed_dim]
        hidden_compressed = hidden_proj.reshape(batch_size, self.compressed_dim, nhid)  # [batch, compressed_dim, nhid]
        hidden_compressed = hidden_compressed.transpose(0, 1)  # [compressed_dim, batch, nhid]
        
        
        key_flat = key_cache.reshape(batch_size, -1)  # [batch, seq_len * nhid]
        key_proj = self.drop(self.tanh(self.key_compress_norm(self.key_compress(key_flat))))  # [batch, nhid * compressed_dim]
        key_compressed = key_proj.reshape(batch_size, self.compressed_dim, nhid)  # [batch, compressed_dim, nhid]
        
        
        value_flat = value_cache.reshape(batch_size, -1)  # [batch, seq_len * nhid]
        value_proj = self.drop(self.tanh(self.value_compress_norm(self.value_compress(value_flat))))  # [batch, nhid * compressed_dim]
        value_compressed = value_proj.reshape(batch_size, self.compressed_dim, nhid)  # [batch, compressed_dim, nhid]
        
        return hidden_compressed, key_compressed, value_compressed
    

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
    
        if initial_cache is not None:
            hidden, key_cache, value_cache = initial_cache
        else:
            # Fallback to fresh cache if none provided
            hidden, key_cache, value_cache = self.init_cache(observation)
            

        emb = self.drop(self.encoder(observation))
        for i in range(seq_len):
            query = self.get_query(emb[i], hidden)
            attn_output,_= self.multihead_attn(query, key_cache, value_cache, temperature, gumbel_softmax, need_weights=False)
            attn_output, query=attn_output.squeeze(1), query.squeeze(1)
            key_cache_i, value_cache_i, hidden_i = self.intermediate_layers(i, emb, query, attn_output, hidden)
            key_cache, value_cache, hidden = self.update_cache(key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i)
        # decoded = self.decoder(hidden[1:])
        decoded = self.decoder(hidden[:self.seq_len])
        cache = self.compress_cache(hidden, key_cache, value_cache)
        return decoded, cache
    
      
            
    def training_step(self, batch, batch_idx):
        # Extract data and targets from batch
        data, targets = batch
        # Initialize cache once per epoch or use existing epoch cache
        if self.epoch_cache is None:
            # Initialize cache once per epoch
            self.epoch_cache = self.init_cache(data)
        else:
            # Detach from computational graph but keep values
            hidden, key_cache, value_cache = self.epoch_cache
            self.epoch_cache = (
                hidden.detach(),
                key_cache.detach(), 
                value_cache.detach()
            )
        
        
        # Forward pass
        output, new_cache = self.forward(
            data, 
            initial_cache= self.epoch_cache, 
            nheads=self.nheads, 
            temperature=self.temperature, 
            gumbel_softmax=self.gumbel_softmax
        )
        
        self.epoch_cache = new_cache
        cache_length = new_cache[1].size(1)  # key_cache sequence length
        
        
        # Reshape outputs and targets for loss computation
        output_flat = output.reshape(-1, output.size(-1))
        targets_flat = targets.reshape(-1)
        
        # Calculate loss
        if self.criterion == 'cross_entropy':
            loss = F.cross_entropy(output_flat, targets_flat)
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")
        
        # Calculate perplexity for logging
        ppl = torch.exp(loss)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_ppl', ppl, prog_bar=True, on_step=True, on_epoch=True)
        self.log('temperature', self.temperature, on_step=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning"""
        # Extract data and targets from batch
        data, targets = batch
        
        # Initialize cache for CBR_RNN
        cache = self.init_cache(data)
        
        # Forward pass
        output, _ = self.forward(
            data, 
            initial_cache=cache, 
            nheads=self.nheads, 
            temperature=self.temperature, 
            gumbel_softmax=self.gumbel_softmax
        )
        
        # Reshape outputs and targets for loss computation
        output_flat = output.reshape(-1, output.size(-1))
        targets_flat = targets.reshape(-1)
        
        # Calculate loss
        if self.criterion == 'cross_entropy':
            loss = F.cross_entropy(output_flat, targets_flat)
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")
        
        # Calculate perplexity for logging
        ppl = torch.exp(loss)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_ppl', ppl, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self):

        if self.optimizer_type == 'sgd':
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