import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from src.language_models.dictionary_corpus import Dictionary, Corpus, tokenize
from src.language_models.model import RNNModel as lstm
from src.language_models.utils import move_to_device
import random
import src.language_models.model as model

# --- Data Loading and Preprocessing ---
def load_data_with_conditions(nounpp_file):
    """Loads data and extracts conditions from nounpp.txt."""

    sentences = []
    gold_data = {}
    condition_map = {}  

    with open(nounpp_file, "r") as f:
        for index, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) >= 5:  
                condition_map[index] = [parts[2], parts[3]]
                sentences.append(parts[1])  
                gold_data[index] = {
                    "correct": parts[1].split()[-1], 
                    "incorrect": parts[4]
                }

    return sentences, gold_data, condition_map

def convert_condition(condition):
  """Converts the condition from [singular, plural] to SS, SP, PP, PS"""
  if condition == ["singular", "singular"]:
    return "SS"
  elif condition == ["singular", "plural"]:
    return "SP"
  elif condition == ["plural", "plural"]:
    return "PP"
  elif condition == ["plural", "singular"]:
    return "PS"
  else:
    return "Unknown"

class SubjectVerbAgreementDataset(Dataset):
    def __init__(self, sentences, gold_data, dictionary, condition_map, condition_to_include):
        self.sentences = sentences
        self.gold_data = gold_data
        self.dictionary = dictionary
        self.condition_map = condition_map
        self.indices = []

        # Filter indices based on conditions
        for index in range(len(sentences)):
            if convert_condition(self.condition_map.get(index)) == condition_to_include:
                self.indices.append(index)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        sentence = self.sentences[index]
        correct_verb = self.gold_data[index]["correct"]
        incorrect_verb = self.gold_data[index]["incorrect"]

        encoded_sentence = [self.dictionary.word2idx.get(word) for word in sentence.split()]
        encoded_correct_verb = self.dictionary.word2idx.get(correct_verb)
        encoded_incorrect_verb = self.dictionary.word2idx.get(incorrect_verb)

        return {
            "sentence": sentence,
            "input_ids": torch.tensor(encoded_sentence),
            "correct_verb": correct_verb,
            "correct_verb_id": torch.tensor(encoded_correct_verb),
            "incorrect_verb": incorrect_verb,
            "incorrect_verb_id": torch.tensor(encoded_incorrect_verb),
            "index": index
        }
        
def evaluate_subject_verb_agreement(model, test_dataloader, device):
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for batch in test_dataloader:#tqdm.tqdm(test_dataloader, desc="Evaluating"):
            outputs = None
            input_ids = batch["input_ids"].transpose(0,1).to(device)
            correct_verb_id = batch["correct_verb_id"]
            incorrect_verb_id = batch["incorrect_verb_id"]
            
            batch_size = input_ids.size(1)
            hidden = model.init_hidden(batch_size)
            hidden =(hidden[0].to(device), hidden[1].to(device))
            
            outputs,_ = model(input_ids, hidden)
            outputs = outputs.transpose(0,1)  
            
            verb_position = input_ids.size(0)-1
            verb_logits = outputs[:,verb_position,:] 
            
            log_probs = torch.nn.functional.log_softmax(verb_logits, dim=-1)

            # correct_logits = []
            # incorrect_logits=[]
            # for i in range(verb_logits.size(0)) : 
            #     correct_logits.append(verb_logits[i,correct_verb_id[i]])
            #     incorrect_logits.append(verb_logits[i,incorrect_verb_id[i]])
            # correct_logits = torch.tensor(correct_logits)
            # incorrect_logits = torch.tensor(incorrect_logits)
            
            
            correct_logits = log_probs.gather(1, correct_verb_id.unsqueeze(1)).squeeze(1)
            incorrect_logits = log_probs.gather(1, incorrect_verb_id.unsqueeze(1)).squeeze(1)
            
            predictions = (correct_logits>incorrect_logits).squeeze().long()
            correct_predictions += predictions.sum().item()
            total_examples += batch_size

    accuracy = correct_predictions / total_examples
    
    return accuracy

batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nounpp_file = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt"
data_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"  # current directory
#data_path = '/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli'
dictionary = Dictionary(data_path)
sentences, gold_data, condition_map = load_data_with_conditions(nounpp_file)
conditions = ["SS", "SP", "PP", "PS"]
checkpoint_path = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/full_check/epoch_40.pt"  # Replace with your checkpoint path
#checkpoint_path = '/scratch2/mrenaudin/colorlessgreenRNNs/data/agreement/English/generated.output_epoch_40.pt'
model = lstm('LSTM', 50001, 200, 650, 2, 0.2, False)
with open(checkpoint_path, 'rb') as f:
    state_dict = torch.load(f, map_location='cuda' if device =='cuda' else 'cpu')
    model.load_state_dict(state_dict)

# import sys
# import src.language_models.model  # Ensure the correct module is imported

# # Redirect 'model' to the correct module
# sys.modules['model'] = src.language_models.model

# with open(checkpoint_path, 'rb') as f:
#     print("Loading the model")
#     if torch.cuda.is_available():
#         model = torch.load(f)
#     else:
#         # to convert model trained on cuda to cpu model
#         model = torch.load(f, map_location=torch.device("cpu"))
#         #model = torch.load(f, map_location = lambda storage, loc: storage)

#model = torch.compile(model)
model.to(device)
model.eval()   
        
for condition in conditions:
    print(f"Evaluating Condition: {condition}")
    test_dataset = SubjectVerbAgreementDataset(sentences, gold_data, dictionary, condition_map, condition)
    random_indices = random.sample(range(len(test_dataset)), 3)
    # for idx in random_indices:
    #     sentence = test_dataset[idx]
    #     print(sentence)
    test_dataloader = DataLoader(test_dataset, batch_size=128)
    accuracy = evaluate_subject_verb_agreement(model, test_dataloader, device)
    print(f"Subject-Verb Agreement Accuracy ({condition}): {accuracy:.4f}\n")