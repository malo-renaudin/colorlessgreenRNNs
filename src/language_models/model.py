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
        # print(output)
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
    def __init__(self, ntoken, ninp, nhid, device, nheads, dropout=0.5):
        super().__init__()
        # same layers as Timkey
        self.device = device
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.score_attn = nn.Softmax(dim=-1)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.q = nn.Linear(ninp + nhid, nhid)
        self.intermediate_h = nn.Linear(nhid * 4, nhid * 4)
        self.decoder = nn.Linear(
            nhid, ntoken
        )  # checker si différence avec ntoken importante
        self.q_norm = torch.nn.LayerNorm(nhid)
        self.int_norm = torch.nn.LayerNorm(nhid * 4)
        self.f_norm = torch.nn.LayerNorm(nhid * 3)
        self.nhid = nhid
        self.attn_div_factor = np.sqrt(nhid)
        self.final_h = nn.Linear(nhid * 4, nhid * 3)
        self.nheads = nheads

        # for multihead attention
        if self.nheads > 1:
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=nhid, num_heads=nheads, batch_first=True
            )

    # same weight initialization as Timkey
    def init_weights(self, freeze_embedding, aux_objective):
        """Initialize encoder and decoder weights"""
        initrange = 0.1
        if not freeze_embedding:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if aux_objective:
            self.aux_decoder.bias.data.fill_(0)
            self.aux_decoder.weight.data.uniform_(-initrange, initrange)

    # def init_hidden(self, bsz):
    #     """Initialize a fresh hidden state"""
    #     weight = next(self.parameters()).data

    #     return torch.tensor(weight.new(bsz, self.nhid).zero_())

    def init_cache(self, observation):
        if len(observation.size()) > 1:
            bsz = observation.size(dim=-1)
        else:
            bsz = 1
        seq_len = observation.size(dim=0)

        return (
            torch.zeros(1, bsz, self.nhid).to(self.device),
            torch.zeros(1, bsz, self.nhid).to(self.device),
            torch.zeros(1, bsz, self.nhid).to(self.device),
        )

    def forward(self, observation, initial_cache, nheads):
        # Get dimensions
        seq_len = observation.size(0)  # if len(observation.size()) > 1 else 1
        observation = observation.to(self.device)
        # Unpack initial cache
        hidden, key_cache, value_cache = (
            initial_cache  # hidden is initialized as the query and updated at each time step
        )

        # 1. Encode observations
        emb = self.drop(self.encoder(observation))
        # Process sequence : is there another more efficient way to compute causal attention than looping ?
        for i in range(
            seq_len
        ):  # need to keep sequential processing as the core structure is recurrent (each new word needs the hidden state obtained after prediction of the last word)
            # 2. Concatenate with previous hidden state
            # multiply hidden[i] by a mask (lower triangular matrix of 1s)
            query = self.drop(
                self.tanh(self.q_norm(self.q(torch.cat((emb[i], hidden[i]), -1))))
            )  # b * d
            query = query.unsqueeze(1)
            if nheads == 1:
                attn_output = scaled_dot_product_attention(
                    query,
                    key_cache.transpose(0, 1),
                    value_cache.transpose(
                        0, 1
                    ),  # batch dimension needs to be the first one, hence the transpose (and the unsuqeeze on the query), second dim is seq len
                    is_causal=True,
                )
                attn = attn_output.squeeze(1)

                intermediate = self.drop(
                    self.tanh(
                        self.int_norm(
                            self.intermediate_h(
                                torch.cat(
                                    (emb[i], query.squeeze(1), attn, hidden[i]), -1
                                )
                            )
                        )
                    )
                )
                key_cache_i, value_cache_i, hidden_i = self.drop(
                    self.tanh(self.f_norm(self.final_h(intermediate)))
                ).split(self.nhid, dim=-1)

                hidden = torch.cat((hidden, hidden_i.unsqueeze(0)), dim=0)
                key_cache = torch.cat((key_cache, key_cache_i.unsqueeze(0)), dim=0)

                value_cache = torch.cat(
                    (value_cache, value_cache_i.unsqueeze(0)), dim=0
                )
            else:
                attn_output = self.multihead_attn(
                    query, key_cache.transpose(0, 1), value_cache.transpose(0, 1)
                )
                # outputs attn output and attn output weights
                attn = attn_output[0]  # .squeeze(1)
                intermediate = self.drop(
                    self.tanh(
                        self.int_norm(
                            self.intermediate_h(
                                torch.cat(
                                    (
                                        emb[i].unsqueeze(0),
                                        query.transpose(0, 1),
                                        attn.transpose(0, 1),
                                        hidden[i].unsqueeze(0),
                                    ),
                                    -1,
                                )
                            )
                        )
                    )
                )

                intermediate_2 = self.drop(
                    self.tanh(self.f_norm(self.final_h(intermediate)))
                )
                key_cache_i, value_cache_i, hidden_i = intermediate_2.split(
                    self.nhid, dim=-1
                )
                hidden = torch.cat((hidden, hidden_i), dim=0)
                key_cache = torch.cat((key_cache, key_cache_i), dim=0)
                value_cache = torch.cat((value_cache, value_cache_i), dim=0)

        decoded = self.decoder(hidden[1:])

        return decoded, hidden
