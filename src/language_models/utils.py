# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import os
import logging
import pandas as pd


def repackage_hidden(h):
    """Detaches hidden states from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, seq_length):
    seq_len = min(seq_length, len(source) - 1 - i)
    data = source[i : i + seq_len]
    # predict the sequences shifted by one word
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    # if device = 'cuda':
    #     #data = data.cuda()
    data = data.to(device)
    return data


def save_checkpoint(model, optimizer, experiment_name, epoch, batch=None):
    """Save model checkpoint."""
    checkpoint_dir = "checkpoints"

    # Create a subfolder for the experiment within the checkpoints directory
    experiment_dir = os.path.join(checkpoint_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    # if batch is None or batch % 10 == 0:

    if batch is None:
        filename = f"{experiment_dir}/epoch_{epoch}.pt"
    else:
        filename = f"{experiment_dir}/epoch_{epoch}_batch_{batch}.pt"
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    logging.info(f"Checkpoint saved: {filename}")


def move_to_device(hidden, device):
    """Move each tensor in the hidden state tuple to the specified device."""
    if isinstance(hidden, torch.Tensor):
        return hidden.to(device)
    else:
        return tuple(move_to_device(h, device) for h in hidden)


def save_val_loss_data(val_loss_data, folder, filename):
    val_loss_df = pd.DataFrame(val_loss_data)
    val_loss_df.to_csv(os.path.join(folder, filename), index=False)


def load_model(
    classmodel,
    model,
    ntokens,
    emsize,
    nhid,
    nheads,
    dropout,
    device,
    nlayers,
    tied,
    checkpoint_path,
):
    import model as m

    if classmodel == "RNNModel":
        model = m.RNNModel(model, ntokens, emsize, nhid, nlayers, dropout, tied)

    elif classmodel == "CBR_RNN":
        model = m.CBR_RNN(ntokens, emsize, nhid, device, nheads, dropout)

    if checkpoint_path:
        with open(checkpoint_path, "rb") as f:
            state_dict = torch.load(
                f, map_location="cuda" if device == "cuda" else "cpu"
            )
            model.load_state_dict(state_dict)

    model = model.to(device)

    return model
