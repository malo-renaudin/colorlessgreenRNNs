# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse

lm_parser = argparse.ArgumentParser(add_help=False)

lm_parser.add_argument("--data", type=str, help="location of the data corpus")
lm_parser.add_argument("--name", type=str, help="experiment name")
lm_parser.add_argument(
    "--classmodel", type=str, help="model class (RNNModel, CBR_RNN, Stack_LSTM)"
)
lm_parser.add_argument(
    "--model",
    type=str,
    default="LSTM",
    help="type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)",
)
lm_parser.add_argument(
    "--emsize", type=int, default=128, help="size of word embeddings"
)
lm_parser.add_argument(
    "--nhid", type=int, default=128, help="number of hidden units per layer"
)
lm_parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
lm_parser.add_argument(
    "--dropout",
    type=float,
    default=0.2,
    help="dropout applied to layers (0 = no dropout)",
)
lm_parser.add_argument(
    "--tied", action="store_true", help="tie the word embedding and softmax weights"
)

lm_parser.add_argument("--lr", type=float, default=20, help="initial learning rate")
lm_parser.add_argument("--clip", type=float, default=1, help="gradient clipping")
lm_parser.add_argument("--epochs", type=int, default=40, help="upper epoch limit")
lm_parser.add_argument(
    "--batch_size", type=int, default=256, metavar="N", help="batch size"
)

lm_parser.add_argument("--bptt", type=int, default=35, help="sequence length")


lm_parser.add_argument("--seed", type=int, default=1111, help="random seed")
lm_parser.add_argument("--cuda", action="store_true", help="use CUDA")
lm_parser.add_argument(
    "--log-interval", type=int, default=1000, metavar="N", help="report interval"
)
lm_parser.add_argument(
    "--save", type=str, default="model.pt", help="path to save the final model"
)
lm_parser.add_argument(
    "--log", type=str, default="log.txt", help="path to logging file"
)
lm_parser.add_argument(
    "--nheads",
    type=int,
    default=1,
    help="number of attention heads in the CBR_RNN model",
)


def checkpoint_path_or_false(arg):
    if arg is None:
        return False
    return arg


lm_parser.add_argument(
    "--checkpoint_path",
    type=checkpoint_path_or_false,
    default=None,
    help="path to a checkpoint of the model",
)
lm_parser.add_argument(
    "--batch_check",
    type=int,
    default=1000,
    help="if batch_check=n, checkpoints will be saved each n batches within an epoch",
)
lm_parser.add_argument(
    "--epoch_checkpointed",
    type=checkpoint_path_or_false,
    default=None,
    help="number of the epoch for input checkpoint",
)
lm_parser.add_argument(
    "--vocab",
    type=str,
    default=None,
    help="path to the vocabulary file",
)

lm_parser.add_argument(
    "--optimizer",
    type=str,
    default="SGD",
    help="optimizer to use (SGD, Adam)",
)

lm_parser.add_argument(
    "--gumbel_softmax",
    action="store_true",
    help="Enable gumbel softmax with tau scheduler"
)

lm_parser.add_argument(
    "--memory_size",
    type=int,
    default=104,
    help='size of the stack in the stack lstm'
)

lm_parser.add_argument(
    '--memory_dim',
    type=int,
    default=5,
    help="dim of stack elements"
)
lm_parser.add_argument(
    '--cell_sparsity_lambda',
    type = float,
    default = 0,
    help = 'lambda for the regularization on cell sparsity'
)
lm_parser.add_argument(
    "--neuron_reg",
    action = 'store_true',
    help ='enable lt st neuron regularization in an RNN'
)
lm_parser.add_argument(
    "--positional_encoding",
    action="store_true",
    help="Enable sinusoidal positional encoding"
)
lm_parser.add_argument(
    "--min_temp",
    type=float,
    default=0.1,
    help="minimum temperature for the gumbel softmax"
)
lm_parser.add_argument(
    '--checkpoint_dir',
    type=str,
    default = 'checkpoints',
    help='directory in which the checkpoints are saved'
)