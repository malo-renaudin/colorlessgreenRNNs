# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch.utils.data.dataloader
from torch.nn.functional import scaled_dot_product_attention
import numpy as np
import logging
import math
#from src.language_models.utils import PositionalEncoding


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.
    ntoken: vocab size
    nip: embedding size
    """

    def __init__(
        self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False
    ):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
            )
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return (
                weight.new(self.nlayers, bsz, self.nhid).zero_(),
                weight.new(self.nlayers, bsz, self.nhid).zero_(),
            )
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()


class CBR_RNN(nn.Module):
    # goal here is to reuse CBR_RNN but with scaled dot product attention for more efficient computations.
    # Also I got rid of options such as loading pretrained embeddings, and ablating attention to simplify the code.
    # In the future if those options are needed, they can still be copy pasted from William's code as the structure hasn't changed
    def __init__(self, ntoken, ninp, nhid, nheads, dropout=0.5, device=None):
        super().__init__()
        # same layers as Timkey
        self.device = device
        self.nheads = nheads
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.score_attn = nn.Softmax(dim=-1)
        self.encoder = nn.Embedding(ntoken, ninp)
        #self.pos_encoder = PositionalEncoding(ninp, bptt)
        self.q = nn.Linear(ninp + nhid, nhid)
        self.intermediate_h = nn.Linear(nhid * 4, nhid * 4)
        self.decoder = nn.Linear(nhid, ntoken)
        self.q_norm = torch.nn.LayerNorm(nhid)
        self.int_norm = torch.nn.LayerNorm(nhid * 4)
        self.f_norm = torch.nn.LayerNorm(nhid * 3)
        self.nhid = nhid
        self.final_h = nn.Linear(nhid * 4, nhid * 3)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=nhid, num_heads=nheads, batch_first=True
        )

        self.init_weights()

    def init_weights(self):
        """Initialize model weights for better training dynamics"""
        # General initialization
        for name, param in self.named_parameters():
            if "weight" in name:
                if "norm" in name:
                    nn.init.ones_(param)
                elif "encoder" in name:
                    nn.init.normal_(param, mean=0, std=0.01)
                elif "decoder" in name:
                    nn.init.normal_(param, mean=0, std=0.01)
                else:
                    # Standard He initialization for processing layers
                    nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="tanh")
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_cache(self, observation, nheads):
        """Initialize hidden state and attention caches with better initialization strategy"""
        if len(observation.size()) > 1:
            bsz = observation.size(dim=-1)
        else:
            bsz = 1

        hidden = torch.zeros(1, bsz, self.nhid).to(self.device) 
        if nheads == 1:
            key_cache = torch.zeros(bsz, 1, 1, self.nhid).to(self.device) 
            value_cache = torch.zeros(bsz, 1, 1, self.nhid).to(self.device) 
        elif nheads>1:
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
    
    def hard_attention_single_head(self,query, key, value, temperature, gumbel_softmax=None, attn_mask=None,dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        if gumbel_softmax: 
            attn_weight = torch.gumbel_softmax(attn_weight, tau=temperature, hard=False, dim=-1)
        else : 
            attn_weight = attn_weight/temperature
            attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=self.training)
        return attn_weight @ value
    
    def hard_attention_multi_head(self,query, key, value, num_heads, temperature, 
                                attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, 
                                enable_gqa=False) -> torch.Tensor:

      
        batch_size, seq_len, embed_dim = query.shape
        head_dim = embed_dim // num_heads
        
        # Reshape for multi-head attention: (batch_size, seq_len, num_heads, head_dim)
        query = query.view(batch_size, seq_len, num_heads, head_dim)
        key = key.view(batch_size, key.size(1), num_heads, head_dim)
        value = value.view(batch_size, value.size(1), num_heads, head_dim)
        
        # Transpose to: (batch_size, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Get dimensions for attention computation
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        
        # Create attention bias
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        
        # Apply causal masking if needed
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias = attn_bias.to(query.dtype)
        
        # Apply custom attention mask if provided
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias
        
        # Handle grouped query attention
        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
        
        # Compute attention weights for all heads
        # query: (batch_size, num_heads, seq_len, head_dim)
        # key: (batch_size, num_heads, seq_len, head_dim)
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias.unsqueeze(0).unsqueeze(0)  # Broadcast to all heads
        
       
        attn_weight = torch.nn.functional.gumbel_softmax(attn_weight, tau=temperature, hard=False, dim=-1)
        
        
        # Apply dropout
        attn_weight = torch.dropout(attn_weight, dropout_p, train=self.training)
        
        # Apply attention to values
        # attn_weight: (batch_size, num_heads, seq_len, seq_len)
        # value: (batch_size, num_heads, seq_len, head_dim)
        attn_output = attn_weight @ value  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, embed_dim)
        
        return attn_output
        
    def attention_layer(self, query, key_cache, value_cache, nheads, temperature, gumbel_softmax):
        if nheads == 1 :
                query = query.unsqueeze(1)
                
                # Ensure all tensors are on the same device
                if query.device != key_cache.device:
                    key_cache = key_cache.to(query.device)
                if query.device != value_cache.device:
                    value_cache = value_cache.to(query.device)
                if gumbel_softmax is False:  
                    try:
                        attn_output = scaled_dot_product_attention(
                            query, key_cache, value_cache, is_causal=False
                        )
                    except Exception as e:
                        logging.error(f"Error in attention computation: {str(e)}")
                        raise
                    attn = attn_output.squeeze(1).squeeze(1)
                    del attn_output  # No longer needed after squeezing
                    query = query.squeeze(1).squeeze(1)
                else:
                    try:
                        attn_output = self.hard_attention_single_head(
                            query, key_cache, value_cache, temperature, is_causal=False
                        )
                    except Exception as e:
                        logging.error(f"Error in attention computation: {str(e)}")
                        raise
                    attn = attn_output.squeeze(1).squeeze(1)
                    del attn_output  # No longer needed after squeezing
                    query = query.squeeze(1).squeeze(1)

            
        else:
            if gumbel_softmax is False : 
                attn_output, _ = self.multihead_attn(
                    query, key_cache, value_cache, is_causal=False
                )
                attn = attn_output.squeeze(1)
                del attn_output  # No longer needed after squeezing
                query = query.squeeze(1)
            else : 
                attn_output = self.hard_attention_multi_head(query, key_cache, value_cache, nheads, temperature)
                attn = attn_output.squeeze(1)
                del attn_output  # No longer needed after squeezing
                query = query.squeeze(1)
            
        return attn, query
    
    def intermediate_layers(self, i, emb, query, attn, hidden):
        intermediate_input = torch.cat((emb[i], query, attn, hidden[-1]), -1)
        del query, attn  
        intermediate = self.drop(
            self.tanh(self.int_norm(self.intermediate_h(intermediate_input)))
        )
        del intermediate_input  
        final_output = self.drop(self.tanh(self.f_norm(self.final_h(intermediate))))
        del intermediate  
        key_cache_i, value_cache_i, hidden_i = final_output.split(self.nhid, dim=-1)
        del final_output
        return key_cache_i, value_cache_i, hidden_i
    
    def get_query(self, emb, hidden):
        combined = torch.cat((emb, hidden[-1]), -1)
        query = self.drop(self.tanh(self.q_norm(self.q(combined))))
        del combined  # No longer needed after creating query
        query = query.unsqueeze(1)
        return query
    
    def forward(self, observation, initial_cache, nheads, temperature, gumbel_softmax):
        seq_len = observation.size(0)
        hidden, key_cache, value_cache = initial_cache
        # 1. Encode observations
        emb = self.drop(self.encoder(observation))
        # if positional_encoding:
        #     emb=self.pos_encoder(emb)
        del observation  # No longer needed after encoding
        
        for i in range(seq_len):
            # 2. Concatenate with previous hidden state
            
            
            query = self.get_query(emb[i], hidden)
            
            attn, query = self.attention_layer(query, key_cache, value_cache, nheads, temperature, gumbel_softmax)
            
            key_cache_i, value_cache_i, hidden_i = self.intermediate_layers(i, emb, query, attn, hidden)
            
            key_cache, value_cache, hidden = self.update_cache(key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i, nheads)
            
            del key_cache_i, value_cache_i, hidden_i  # No longer needed after concatenation

        decoded = self.decoder(hidden[1:])

        return decoded, hidden


class Stack_LSTM(nn.Module):
    
    def __init__(self, vocab_size, embsz, hidden_dim, device,n_layers=2, memory_size=104, memory_dim = 5):
        super(Stack_LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        self.lstm = nn.LSTM(embsz, self.hidden_dim, self.n_layers)

        self.W_y = nn.Linear(self.hidden_dim, vocab_size)
        self.W_n = nn.Linear(self.hidden_dim, self.memory_dim)
        self.W_a = nn.Linear(self.hidden_dim, 2)
        self.W_sh = nn.Linear (self.memory_dim, self.hidden_dim)

        self.E    = nn.Embedding(vocab_size,embsz)
        # Actions -- push : 0 and pop: 1
        self.softmax = nn.Softmax(dim=1) 
        self.sigmoid = nn.Sigmoid ()
        self.device  = device
        

        

        
    def init_hidden (self, bsz):
        return (torch.zeros (self.n_layers, bsz, self.hidden_dim).to(self.device),
                torch.zeros (self.n_layers, bsz, self.hidden_dim).to(self.device))
    def init_stack(self, bsz):
        return torch.zeros (bsz, self.memory_size,self.memory_dim).to(self.device)
    
    def forward(self, input, hidden0, stack, temperature=1.):
        """
        Forward pass of the LSTM with stack updates after each word
        
        Args:
            input: Input tensor of shape [seq_len, batch_size]
            hidden0: Initial hidden state (h0, c0) where each has shape [n_layers, batch_size, hidden_dim]
            stack: Initial stack of shape [batch_size, memory_length, memory_dim]
            temperature: Temperature for softmax (default=1.0)
            
        Returns:
            output: Output logits of shape [seq_len, batch_size, vocab_size]
            hidden: Final hidden state (h, c)
            stack: Final stack state [batch_size, memory_length, memory_dim]
        """
        seqlen, batch_size = input.size()
        
        h0, c0 = hidden0
        h0 = h0.clone()
        
        
        stack_top = stack[:, 0, :]  # [batch_size, memory_dim]
        transformed_stack_top = self.W_sh(stack_top)  # [batch_size, hidden_dim]
        
        # Add to hidden state - using broadcasting to apply to all layers
        expanded_stack_top = transformed_stack_top.unsqueeze(0).repeat(self.n_layers, 1, 1).contiguous()
        hidden0_modified = h0 + expanded_stack_top  # [n_layers, batch_size, hidden_dim]
        
        # Embed the input sequence
        embedded = self.E(input)  # [seq_len, batch_size, embsz]
        
        # Process each word in the sequence and update the stack after each word
        current_h = hidden0_modified
        current_c = c0
        hidden = (current_h, current_c)
        
        # We need to store outputs for each timestep
        outputs = []
        
        for i in range(seqlen):
            # Process current word through LSTM
            # embedded[i] has shape [batch_size, embsz]
            # We reshape it for processing through LSTM
            current_word = embedded[i].unsqueeze(0)  # [1, batch_size, embsz]
            
            # Process through LSTM
            # ht will have shape [1, batch_size, hidden_dim]
            ht, hidden = self.lstm(current_word, hidden)
            
            # Squeeze out the singleton dimension
            ht = ht.squeeze(0)  # [batch_size, hidden_dim]
            
            # Get output for vocabulary prediction
            current_output = self.W_y(ht)  # [batch_size, vocab_size]
            outputs.append(current_output.unsqueeze(0))  # Add to our list of outputs
            
            # Compute stack action probabilities
            action_logits = self.W_a(ht)  # [batch_size, 2]
            action_weights = self.softmax(action_logits / temperature)  # [batch_size, 2]
            
            # Compute new element to potentially push to stack
            new_elt = self.sigmoid(self.W_n(ht))  # [batch_size, memory_dim]
            new_elt_expanded = new_elt.unsqueeze(1)  # [batch_size, 1, memory_dim]
            
            # Prepare push operation - move everything down and put new element at top
            push_side = torch.cat([new_elt_expanded, stack[:, :-1, :]], dim=1)
            
            # Prepare pop operation - move everything up and add a zero at the bottom
            # This is creating a tensor of zeros to fill the bottom of the stack after popping
            zeros = torch.zeros(batch_size, 1, self.memory_dim, device=self.device)
            pop_side = torch.cat([stack[:, 1:, :], zeros], dim=1)
            
            # Apply weighted action - weight of each action determines how much we push vs pop
            push_weight = action_weights[:, 0].reshape(batch_size, 1, 1)
            pop_weight = action_weights[:, 1].reshape(batch_size, 1, 1)
            
            # Update the stack as weighted combination of push and pop operations
            stack = push_weight * push_side + pop_weight * pop_side
            
            # Update hidden state for next iteration with new stack information
            if i < seqlen - 1:  # Only update if not the last timestep
                stack_top = stack[:, 0, :]  # Get new stack top
                transformed_stack_top = self.W_sh(stack_top)  # Transform to hidden dimension
                
                # Instead of just updating hidden0_modified, we need to update the actual hidden state
                # We need to add the stack information to all layers of the hidden state
                expanded_stack_top = transformed_stack_top.unsqueeze(0).repeat(self.n_layers, 1, 1)
                current_h = hidden[0] + expanded_stack_top
                current_c = hidden[1]  # Keep cell state the same
                hidden = (current_h, current_c)
        
        # Concatenate all outputs from the sequence
        output = torch.cat(outputs, dim=0)  # [seq_len, batch_size, vocab_size]
        
        return output, hidden, stack