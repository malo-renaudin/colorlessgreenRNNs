import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from src.language_models.dictionary_corpus import Dictionary, Corpus, tokenize
from src.language_models.model import RNNModel as lstm
from src.language_models.utils import move_to_device
import random
import pandas as pd
from collections import defaultdict
import numpy as np

batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"  # current directory
dictionary = Dictionary(data_path)
checkpoint_path = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/full_check/epoch_40.pt"  # Replace with your checkpoint path

model = lstm('LSTM', 50001, 200, 650, 2, 0.2, False)
with open(checkpoint_path, 'rb') as f:
    state_dict = torch.load(f, map_location='cuda' if device =='cuda' else 'cpu')
    model.load_state_dict(state_dict)
model.to(device)
model.eval() 

nounpp = '//scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt'

class NounPPDataset(Dataset):
    def __init__(self, nounpp_file, dictionary):
        self.sentences = []
        self.conditions = []
        self.correct = []
        self.wrong = []
        self.encoded_sentences=[]
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
                encoded_sentence = [self.dictionary.word2idx.get(word, self.dictionary.word2idx.get("<unk>")) for word in sentence]
                encoded_correct = self.dictionary.word2idx.get(correct, self.dictionary.word2idx.get("<unk>"))
                encoded_wrong = self.dictionary.word2idx.get(wrong, self.dictionary.word2idx.get("<unk>"))
                
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
            "encoded_sentence": torch.tensor(self.encoded_sentences[idx], dtype=torch.long),
            "correct": self.correct[idx],
            "encoded_correct":torch.tensor(self.encoded_correct[idx], dtype = torch.long),
            "wrong":self.wrong[idx],
            "encoded_wrong":torch.tensor(self.encoded_wrong[idx], dtype = torch.long),
            "condition": self.conditions[idx],
        }
        
test_dataset = NounPPDataset(nounpp, dictionary)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

init_sentence = " ".join(["In service , the aircraft was operated by a crew of five and could accommodate either 30 paratroopers , 32 <unk> and 28 sitting casualties , or 50 fully equipped troops . <eos>",
                    "He even speculated that technical classes might some day be held \" for the better training of workmen in their several crafts and industries . <eos>",
                    "After the War of the Holy League in 1537 against the Ottoman Empire , a truce between Venice and the Ottomans was created in 1539 . <eos>",
                    "Moore says : \" Tony and I had a good <unk> and off-screen relationship , we are two very different people , but we did share a sense of humour \" . <eos>",
                    "<unk> is also the basis for online games sold through licensed lotteries . <eos>"])

def feed_input(model, hidden, w):
    inp = torch.autograd.Variable(torch.LongTensor([[dictionary.word2idx[w]]]))
    out, hidden = model(inp, hidden)
    return out, hidden
def feed_sentence(model, h, sentence):
    outs = []
    for w in sentence:
        out, h = feed_input(model, h, w)
        outs.append(torch.nn.functional.log_softmax(out[0]).unsqueeze(0))
    return outs, h

model.eval()
hidden = model.init_hidden(1) 
init_out, init_h = feed_sentence(model, hidden, init_sentence.split(" "))

condition_accuracies = defaultdict(int)
condition_counts = defaultdict(int)

s = torch.nn.LogSoftmax(dim=-1)
model.eval()
#Forward pass with hidden state update word by word
with torch.no_grad():
    for batch in test_dataloader:
        written = batch['sentence']
        sentence = batch['encoded_sentence']
        correct = batch['encoded_correct']
        wrong = batch['encoded_wrong']
        condition = batch['condition']
        batch_size = sentence.size(0)
        hidden = (init_h[0].expand(-1, batch_size, -1).contiguous(), 
          init_h[1].expand(-1, batch_size, -1).contiguous()) 

        for w in range(sentence.shape[1]):#update hidden state word by word
            word = torch.autograd.Variable(sentence[:,w].unsqueeze(0))
            
            out, hidden = model(word, hidden)
  
        log_probs = s(out.squeeze(0))
        correct_log_probs = log_probs[torch.arange(batch_size), correct]  # Shape: [512]
        print(correct_log_probs)
        wrong_log_probs = log_probs[torch.arange(batch_size), wrong]
        print(wrong_log_probs)
        correct_predictions = np.sum(correct_log_probs >= wrong_log_probs)
     
        for i in range(batch_size):
            cond = condition[i]
            pred = correct_predictions[i].item()  # Extract the boolean value
            condition_counts[cond] += 1
            condition_accuracies[cond] += pred
final_accuracies = {cond: condition_accuracies[cond] / condition_counts[cond] for cond in condition_accuracies}


# Print results
for cond, acc in final_accuracies.items():
    print(f"{cond}: Accuracy = {acc * 100:.2f}%")
