import torch
import os
from torch.utils.data import Dataset
from collections import defaultdict

class NounPPDataset(Dataset):
    def __init__(self, nounpp_file, dictionary):
        self.sentences = []
        self.conditions = []
        self.correct = []
        self.wrong = []
        self.encoded_sentences = []
        self.encoded_correct = []
        self.encoded_wrong = []
        self.dictionary = dictionary

        with open(nounpp_file, "r") as f:
            for line in f:
                line = line.split()
                sentence = line[1:7]
                condition = " ".join(line[7:9])
                wrong = line[9]
                correct = line[6]
                encoded_sentence = [
                    self.dictionary.word2idx.get(
                        word, self.dictionary.word2idx.get("<unk>")
                    )
                    for word in sentence
                ]
                encoded_correct = self.dictionary.word2idx.get(
                    correct, self.dictionary.word2idx.get("<unk>")
                )
                encoded_wrong = self.dictionary.word2idx.get(
                    wrong, self.dictionary.word2idx.get("<unk>")
                )

                self.sentences.append(sentence)
                self.conditions.append(condition)
                self.correct.append(correct)
                self.wrong.append(wrong)
                self.encoded_sentences.append(encoded_sentence)
                self.encoded_correct.append(encoded_correct)
                self.encoded_wrong.append(encoded_wrong)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "encoded_sentence": torch.tensor(
                self.encoded_sentences[idx], dtype=torch.long
            ),
            "correct": self.correct[idx],
            "encoded_correct": torch.tensor(
                self.encoded_correct[idx], dtype=torch.long
            ),
            "wrong": self.wrong[idx],
            "encoded_wrong": torch.tensor(self.encoded_wrong[idx], dtype=torch.long),
            "condition": self.conditions[idx],
        }

def collate_fn(batch):
    """Custom collate function to properly handle sentences as lists of strings."""
    sentences = [item["sentence"] for item in batch]  # Keep lists of words as they are
    encoded_sentences = torch.stack(
        [item["encoded_sentence"] for item in batch]
    )  # Stack tensors
    encoded_correct = torch.stack([item["encoded_correct"] for item in batch])
    encoded_wrong = torch.stack([item["encoded_wrong"] for item in batch])
    correct = [item["correct"] for item in batch]
    wrong = [item["wrong"] for item in batch]
    conditions = [item["condition"] for item in batch]

    return {
        "sentence": sentences,
        "encoded_sentence": encoded_sentences,
        "correct": correct,
        "encoded_correct": encoded_correct,
        "wrong": wrong,
        "encoded_wrong": encoded_wrong,
        "condition": conditions,
    }

def feed_input(model, hidden, w, dictionary, device):
    inp = torch.autograd.Variable(
        torch.LongTensor([[dictionary.word2idx[w]]]).to(device)
    )
    out, hidden = model(inp, hidden)
    return out, hidden


def feed_sentence(model, h, sentence, dictionary, device):
    outs = []
    for w in sentence:
        out, h = feed_input(model, h, w, dictionary, device)
        outs.append(torch.nn.functional.log_softmax(out[0]).unsqueeze(0))
    return outs, h

def eval(model, test_dataloader, init_sentence, dictionary, device):
    condition_accuracies = defaultdict(int)
    condition_counts = defaultdict(int)
    correct_pred = 0
    sentence_details = []

    model.eval()

    hidden = model.init_hidden(1)
    init_out, init_h = feed_sentence(model, hidden, init_sentence.split(" "), dictionary, device)

    with torch.no_grad():
        for batch in test_dataloader:
            out = None
            written = batch["sentence"]
            sentence = batch["encoded_sentence"].to(device)
            correct = batch["encoded_correct"].to(device)
            wrong = batch["encoded_wrong"].to(device)
            condition = batch["condition"]
            batch_size = sentence.size(0)
            hidden = (
                init_h[0].expand(-1, batch_size, -1).contiguous(),
                init_h[1].expand(-1, batch_size, -1).contiguous(),
            )
            for w in range(sentence.shape[1] - 1):
                word = torch.autograd.Variable(sentence[:, w].unsqueeze(0)).to(device)
                out, hidden = model(word, hidden)

            log_probs = torch.nn.functional.log_softmax(out, dim=-1)
            correct_log_probs = log_probs[0, torch.arange(batch_size), correct]
            wrong_log_probs = log_probs[0, torch.arange(batch_size), wrong]
            correct_predictions = correct_log_probs >= wrong_log_probs
            for i in range(batch_size):
                cond = condition[i]
                pred = correct_predictions[i].item()
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

def ablate_neuron(model, l, n, num_neurons):
    gate_indices = torch.tensor(
                [
                    n,
                    n + num_neurons,
                    n + num_neurons * 2,
                    n + num_neurons * 3,
                ]
            )
    if l == 0:
        # Ablate layer 0 neuron
        model.rnn.weight_ih_l0[gate_indices] = 0
        model.rnn.weight_hh_l0[gate_indices] = 0
        model.rnn.bias_ih_l0[gate_indices] = 0
        model.rnn.bias_hh_l0[gate_indices] = 0
    elif l == 1:
        # Ablate layer 1 neuron
        model.rnn.weight_ih_l1[gate_indices] = 0
        model.rnn.weight_hh_l1[gate_indices] = 0
        model.rnn.bias_ih_l1[gate_indices] = 0
        model.rnn.bias_hh_l1[gate_indices] = 0
    return model, gate_indices

def restore_neuron(model, l, gate_indices, weights):
    if l == 0:
        model.rnn.weight_ih_l0[gate_indices] = weights["weight_ih_l0"][gate_indices]    
        model.rnn.weight_hh_l0[gate_indices] = weights["weight_hh_l0"][gate_indices]
        model.rnn.bias_ih_l0[gate_indices] = weights["bias_ih_l0"][gate_indices]
        model.rnn.bias_hh_l0[gate_indices] = weights["bias_hh_l0"][gate_indices]
    elif l == 1:
        model.rnn.weight_ih_l1[gate_indices] = weights["weight_ih_l1"][gate_indices]
        model.rnn.weight_hh_l1[gate_indices] = weights["weight_hh_l1"][gate_indices]
        model.rnn.bias_ih_l1[gate_indices] = weights["bias_ih_l1"][gate_indices]
        model.rnn.bias_hh_l1[gate_indices] = weights["bias_hh_l1"][gate_indices]
    return model

def cache_weights(model):
    weights = {
    "weight_ih_l0": model.rnn.weight_ih_l0.clone(),
    "weight_hh_l0": model.rnn.weight_hh_l0.clone(),
    "bias_ih_l0": model.rnn.bias_ih_l0.clone(),
    "bias_hh_l0": model.rnn.bias_hh_l0.clone(),
    "weight_ih_l1": model.rnn.weight_ih_l1.clone(),
    "weight_hh_l1": model.rnn.weight_hh_l1.clone(),
    "bias_ih_l1": model.rnn.bias_ih_l1.clone(),
    "bias_hh_l1": model.rnn.bias_hh_l1.clone(),
}
    return weights

def evaluate_checkpoint(model, test_loader, init_sentence, dictionary, device, eval, layer=2, num_neurons=650):
    results = {}
    original_accuracies = eval(model, test_loader, init_sentence, dictionary, device)
    results["original"] = original_accuracies

    ####caching
    weights= cache_weights(model)

    for l in range(layer):
        for n in range(num_neurons):
            with torch.no_grad():
                
                model, gate_indices = ablate_neuron(model, l, n, num_neurons)
                res = eval(model, test_loader, init_sentence, dictionary, device)
                results[f"layer_{l}_neuron_{n}"] = res
                #restore weights
                model = restore_neuron(model, l, gate_indices, weights)
    return results
        