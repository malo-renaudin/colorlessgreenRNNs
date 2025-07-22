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
                 optimizer_type='adam', weight_decay=0.0, scheduler_type=None, cache_strategy='epoch'):
        super().__init__()
        
        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters()
        self.cache_strategy = cache_strategy  # either "per_batch" or "per_epoch"
        self.epoch_cache = None
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
            embed_dim=nhid, num_heads=nheads, batch_first=True
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

    def init_cache(self, observation):
        """Initialize hidden state and attention caches"""
        if len(observation.size()) > 1:
            bsz = observation.size(dim=-1)
        else:
            bsz = 1

        hidden = torch.zeros(1, bsz, self.nhid).to(self.device) 
        key_cache = torch.zeros(bsz, 1, self.nhid).to(self.device) 
        value_cache = torch.zeros(bsz, 1, self.nhid).to(self.device) 
        return hidden, key_cache, value_cache

    def update_cache(self, key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i):
        hidden_i = hidden_i.unsqueeze(0)
        hidden = torch.cat((hidden, hidden_i), dim=0)
        key_cache_i = key_cache_i.unsqueeze(1)
        value_cache_i = value_cache_i.unsqueeze(1)
        key_cache = torch.cat((key_cache, key_cache_i), dim=1)
        value_cache = torch.cat((value_cache, value_cache_i), dim=1)
            
        return key_cache, value_cache, hidden

    

    def intermediate_layers(self, i, emb, query, attn, hidden):
        print('ok')
        print('emb[i]', emb[i].shape)
        print('query', query.shape)
        print('attn', attn.shape)
        print('hidden[-1]', hidden[-1].shape)
        intermediate_input = torch.cat((emb[i], query, attn, hidden[-1]), -1)#ici les dernières dimensions sont additionées
        print('intermediate_input', intermediate_input.shape)
        intermediate = self.drop(
            self.tanh(self.int_norm(self.intermediate_h(intermediate_input)))#ici il faut que cette somme soit de 1024. Ca contraint ninp=nhid
        )
        print('intermediat', intermediate.shape)
        final_output = self.drop(self.tanh(self.f_norm(self.final_h(intermediate))))
        print('final_output', final_output.shape)
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
            
        print('hidden', hidden.shape)
        print('key_cache', key_cache.shape)
        print('value_cache', value_cache.shape)   
        emb = self.drop(self.encoder(observation))
        print('emb', emb.shape)
        for i in range(seq_len):
            query = self.get_query(emb[i], hidden)
            print('query', query.shape)
            attn_output,_= self.multihead_attn(query, key_cache, value_cache, temperature, gumbel_softmax, need_weights=False)
            attn_output, query=attn_output.squeeze(1), query.squeeze(1)
            print('attn_output',attn_output.shape)
            key_cache_i, value_cache_i, hidden_i = self.intermediate_layers(i, emb, query, attn_output, hidden)
            print('key_cache_i',key_cache_i.shape)
            print('value_cache_i',value_cache_i.shape)
            print('hidden_i', hidden_i.shape)
            key_cache, value_cache, hidden = self.update_cache(key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i)
            print('key_cache',key_cache.shape)
            print('value_cache',value_cache.shape)
            print('hidden', hidden.shape)
        decoded = self.decoder(hidden[1:])
        print('decoded', decoded.shape)
        return decoded, (hidden, key_cache, value_cache)
    
    def on_train_epoch_start(self):
        """Reset cache at the start of each epoch"""
        print(f"🔄 Starting new epoch - cache strategy: {self.cache_strategy}")
        if self.cache_strategy == "epoch":
            self.epoch_cache = None
            print("✅ Epoch cache reset to None")
            
            
    def training_step(self, batch, batch_idx):
        # Extract data and targets from batch
        data, targets = batch
        
        if self.cache_strategy == "batch":
            # Fresh cache for each batch
            cache = self.init_cache(data)
            print(f"📦 Batch {batch_idx}: Using fresh cache")
            
        elif self.cache_strategy == "epoch":
            if self.epoch_cache is None:
                # Initialize cache once per epoch
                self.epoch_cache = self.init_cache(data)
                print(f"🎯 Batch {batch_idx}: Initialized epoch cache")
            else:
                # Detach from computational graph but keep values
                hidden, key_cache, value_cache = self.epoch_cache
                self.epoch_cache = (
                    hidden.detach(),
                    key_cache.detach(), 
                    value_cache.detach()
                )
                print(f"🔗 Batch {batch_idx}: Using persistent epoch cache (detached)")
            
            cache = self.epoch_cache
        # Forward pass
        output, new_cache = self.forward(
            data, 
            initial_cache=cache, 
            nheads=self.nheads, 
            temperature=self.temperature, 
            gumbel_softmax=self.gumbel_softmax
        )
        if self.cache_strategy == "epoch":
            self.epoch_cache = new_cache
            cache_length = new_cache[1].size(1)  # key_cache sequence length
            print(f"📈 Cache length after batch {batch_idx}: {cache_length}")
        print('output', output.shape)
        print('targets', targets.shape)
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