import sys
import os

sys.path.append('colorlessgreenRNNs')

from src.language_models import model as m
import torch
from evaluation_notebooks.utils import NounPPDataset, collate_fn_nounpp
from src.language_models.dictionary_corpus import Dictionary
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import argparse



def eval_nounpp(checkpoint, temperature, hidden_dim, nheads, data_path, nounpp, device):
    
    dictionary = Dictionary(data_path)
    test_dataset = NounPPDataset(nounpp, dictionary)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, collate_fn=collate_fn_nounpp)
    
    model = m.CBR_RNN(50001, hidden_dim, hidden_dim, nheads, 0.5, device)
    model=model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    condition_accuracies = defaultdict(int)
    condition_counts = defaultdict(int)
    sentence_details = []
    model.eval()
    # Forward pass with hidden state update word by word
    with torch.no_grad():
        for batch in test_dataloader:
            out = None
            written = batch["sentence"]
            sentence = batch["encoded_sentence"]
            correct = batch["encoded_correct"]
            wrong = batch["encoded_wrong"]
            condition = batch["condition"]
            batch_size = sentence.size(0)

            sent = sentence[:, :5].transpose(0, 1).to(device)
            cache = model.init_cache(sent,1)  # regarder si on peut mettre du priming
            # for i in range(sent.shape[1]):
            out, cache = model(sent, cache, 1, temperature, True)
            log_probs = torch.nn.functional.log_softmax(
                out, dim=-1
            )  # s(out.squeeze(0))
            # déja sur correct et wrong log probs, pas les même résultats que sur extract_predictions.py
            correct_log_probs = log_probs[
                -1, torch.arange(batch_size), correct
            ]  # Shape: [512]
            wrong_log_probs = log_probs[-1, torch.arange(batch_size), wrong]
            correct_predictions = correct_log_probs >= wrong_log_probs

            for i in range(batch_size):
                cond = condition[i]
                pred = correct_predictions[i].item()  # Convert tensor to Python boolean
                condition_counts[cond] += 1
                condition_accuracies[cond] += pred

                sentence_details.append(
                    {
                        "sentence": written[i],
                        "condition": condition[i],
                        "correct_log_prob": correct_log_probs[i],
                        "wrong_log_prob": wrong_log_probs[i],
                        "model_prefers_correct": pred,
                    } 
                )

    final_accuracies = {
        cond: condition_accuracies[cond] / condition_counts[cond]
        for cond in condition_accuracies
    }
    return final_accuracies


