import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import string
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np 

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

class BLiMPDataset(Dataset):
    def __init__(self, blimp_subset, dictionary):
        self.dataset = load_dataset('nyu-mll/blimp', blimp_subset, split = 'train')
        self.dictionary = dictionary
        self.encoded_pairs = []

        for example in self.dataset:
            sentence_good = example['sentence_good']
            sentence_bad = example['sentence_bad']
            sentence_good = sentence_good.rstrip(string.punctuation)
            sentence_bad = sentence_bad.rstrip(string.punctuation)
            encoded_good = [self.dictionary.word2idx.get(word, self.dictionary.word2idx.get("<unk>")) for word in sentence_good.split()]
            #encoded_good = [self.dictionary.word2idx.get(word) for word in sentence_good.split()]
            encoded_bad = [self.dictionary.word2idx.get(word, self.dictionary.word2idx.get("<unk>")) for word in sentence_bad.split()]
            #encoded_bad = [self.dictionary.word2idx.get(word) for word in sentence_bad.split()]
            self.encoded_pairs.append({
                "sentence_good": sentence_good,
                "sentence_bad": sentence_bad,
                "encoded_good": torch.tensor(encoded_good, dtype=torch.long),
                "encoded_bad": torch.tensor(encoded_bad, dtype=torch.long),
            })

    def __len__(self):
        return len(self.encoded_pairs)

    def __getitem__(self, idx):
        return self.encoded_pairs[idx]
 
 
    
def collate_fn_nounpp(batch):
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

def collate_fn_blimp(batch):
    encoded_good_sequences = [item['encoded_good'] for item in batch]
    encoded_bad_sequences = [item['encoded_bad'] for item in batch]
    sentence_good = [item['sentence_good'] for item in batch]
    sentence_bad = [item['sentence_bad'] for item in batch]
    return {
        'sentence_bad':sentence_bad,
        'sentence_good': sentence_good,
        'encoded_good': pad_sequence(encoded_good_sequences, batch_first=True),
        'encoded_bad': pad_sequence(encoded_bad_sequences, batch_first=True)
    }
    
def feed_input(model, hidden, w, dictionary, device):
    inp = torch.autograd.Variable(
        torch.LongTensor([[dictionary.word2idx[w]]]).to(device)
    )
    out, hidden= model(inp, hidden)
    return out, hidden


def feed_sentence(model, h, sentence, dictionary, device):
    outs = []
    for w in sentence:
        out, h = feed_input(model, h, w, dictionary, device)
        outs.append(torch.nn.functional.log_softmax(out[0]).unsqueeze(0))
    return outs, h 

def eval_cbr_nounpp(model, test_dataloader, temperature, nheads, gumbel_softmax):
    condition_accuracies = defaultdict(int)
    condition_counts = defaultdict(int)
    correct_pred = 0
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

            sent = sentence[:, :5].transpose(0, 1)
            cache = model.init_cache(sent,1)  # regarder si on peut mettre du priming
            # for i in range(sent.shape[1]):
            out, cache = model(sent, cache, nheads, temperature, gumbel_softmax)
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

def compute_seq_nll(model,data, hidden, batch_size):
    batch_size, seq_len = data.shape
    #mask
    mask = (data!=0).float()
    #forward pass
    
    pad = data.swapaxes(0,1)
    
    output, hidden = model(pad, hidden)
    #target
    targets = data[:, 1:]#.swapaxes(0,1)
    #log probs
    log_probs = F.log_softmax(output, dim=-1)
    log_probs = log_probs[:-1]
    log_probs = log_probs.swapaxes(0,1)
    #nll loss
    #WITH NLL LOSS
    nll_loss = F.nll_loss(
            log_probs.reshape(-1, log_probs.size(-1)),
            targets.reshape(-1),
            reduction='none'
        )#.reshape(batch_size, max_len - 1)
    nll_loss=nll_loss.reshape(batch_size, seq_len - 1)
    #mask loss
    masked_nll_loss = nll_loss * mask[:, 1:]
    # Sum the negative log-likelihood over the sequence for each example
    sequence_nll = masked_nll_loss.sum(dim=1)
    return -sequence_nll

def move_to_device(hidden, device):
    """Move each tensor in the hidden state tuple to the specified device."""
    if isinstance(hidden, torch.Tensor):
        return hidden.to(device)
    else:
        return tuple(move_to_device(h, device) for h in hidden)


def eval_lstm_blimp(model, test_dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    #Forward pass with hidden state update word by word
    with torch.no_grad():
        for batch in test_dataloader:

            sentence_good = batch['sentence_good']
            sentence_bad = batch['sentence_bad']

            good = batch['encoded_good'].to(device)
            bad = batch['encoded_bad'].to(device)
        
            batch_size = good.size(0)

            hidden_good = model.init_hidden(batch_size)
            hidden_good = move_to_device(hidden_good, device)
            hidden_bad = model.init_hidden(batch_size) 
            hidden_bad=move_to_device(hidden_bad,device)       
            seq_nll_good = compute_seq_nll(model,good, hidden_good, batch_size)
            seq_nll_bad = compute_seq_nll(model,bad, hidden_bad, batch_size)
            predictions = (seq_nll_good > seq_nll_bad).cpu().numpy()
            correct_predictions += np.sum(predictions)
            total_predictions += batch_size
      
            
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy on {test_dataloader.dataset.dataset.config_name}: {accuracy * 100:.2f}%")
    return accuracy


def eval_lstm_nounpp(model, test_dataloader, init_sentence, dictionary, device):
    condition_accuracies = defaultdict(int)
    condition_counts = defaultdict(int)
    correct_pred = 0
    sentence_details = []

    model.eval()

    hidden = model.init_hidden(1)
    #stack = model.init_stack(1)
    init_out, init_h= feed_sentence(model, hidden, init_sentence.split(" "), dictionary, device)
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
            hidden = move_to_device(hidden, device)
            # stack = (
            #     init_stack.expand(batch_size, -1, -1).contiguous(),
            # )
            # stack=stack[0]
            for w in range(sentence.shape[1] - 1):

                word = torch.autograd.Variable(sentence[:, w].unsqueeze(0))
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
