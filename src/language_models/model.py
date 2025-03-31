# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch.utils.data.dataloader
from torch.nn.functional import scaled_dot_product_attention


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.
        ntoken: vocab size
        nip: embedding size
    """

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
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
        #print(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                    weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()

class CBR_RNN(nn.module): 
# goal here is to reuse CBR_RNN but with scaled dot product attention for more efficient computations. 
# Also I got rid of options such as loading pretrained embeddings, and ablating attention to simplify the code.
# In the future if those options are needed, they can still be copy pasted from William's code as the structure hasn't changed
    def __init__(self, ntoken, ninp, nhid, dropout=0.5, tie_weights=False, embedding_file=None, device=None):
        super().__init__()
        #same layers as Timkey
        self.device = device
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.score_attn = nn.Softmax(dim=-1)
        self.encoder = nn.Embedding(ntoken+1, ninp)
        self.q = nn.Linear(ninp+nhid,nhid)
        self.intermediate_h = nn.Linear(nhid*4,nhid*4)
        self.decoder = nn.Linear(nhid, ntoken+1)
        self.q_norm = torch.nn.LayerNorm(nhid)
        self.int_norm = torch.nn.LayerNorm(nhid * 4)
        self.f_norm = torch.nn.LayerNorm(nhid * 3)  
        self.nhid = nhid
        self.attn_div_factor = np.sqrt(nhid)
        
        
    #same weight initialization as Timkey
    def init_weights(self, freeze_embedding, aux_objective):
        """ Initialize encoder and decoder weights """
        initrange = 0.1
        if not freeze_embedding:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if(aux_objective):
            self.aux_decoder.bias.data.fill_(0)
            self.aux_decoder.weight.data.uniform_(-initrange, initrange)

    
    

    def forward(self, observation, initial_cache, attention_mask=None):
        # Get dimensions
        seq_len, batch_size = observation.size(0), observation.size(1) if len(observation.size()) > 1 else 1
        # Unpack initial cache
        hidden, key_cache, value_cache = initial_cache
        # 1. Encode observations
        emb = self.drop(self.encoder(observation))
        # Process sequence : is there another more efficient way to compute causal attention than looping ?
        for i in range(seq_len):
            # 2. Concatenate with previous hidden state
            query = self.drop(self.tanh(self.q_norm(self.q(torch.cat((emb[i],hidden[i]), -1))))) #b * d
            query_n = query.unsqueeze(-1) #b * n * 1
            
            # 3. Attention mechanism
            # Prepare query, key, and value tensors
            k = all_keys[:i+1].permute(1, 0, 2)  # [batch_size, i+1, nhid]
            v = all_values[:i+1].permute(1, 0, 2)  # [batch_size, i+1, nhid]
            
            # Create attention mask if provided
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask[:, i, :i+1]
            
            # Apply attention
            attn_output = scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            
            attn_output = attn_output.squeeze(1)  # [batch_size, nhid]
            
            # 4. Feed-forward (combine attention with query)
            combined_attn = self.dropout(attn_output + query)  # Residual connection
            
            # 5. Split to key, value, hidden
            outputs = self.dropout(self.activation(self.ff2(combined_attn)))
            outputs = self.norm3(outputs)
            
            # Split into key, value, hidden
            key_i, value_i, hidden_i = outputs.chunk(3, dim=-1)
            
            # Store results
            all_hidden[i+1] = hidden_i
            all_keys[i+1] = key_i
            all_values[i+1] = value_i
        
        # Compute outputs
        output_hidden = all_hidden[1:]  # Remove initial state
        decoded = self.decoder(output_hidden)
        
        # Prepare final state for next iteration
        final_cache = (
            all_hidden[-1:],  # Just the last hidden state
            all_keys[-1:],    # Just the last key
            all_values[-1:]   # Just the last value
        )
        
        return decoded, final_cache
    
    def init_cache(self, batch_size=1):
        """Initialize cache for a new sequence"""
        return (
            torch.zeros(1, batch_size, self.nhid, device=self.device),
            torch.zeros(1, batch_size, self.nhid, device=self.device), 
            torch.zeros(1, batch_size, self.nhid, device=self.device)
        )