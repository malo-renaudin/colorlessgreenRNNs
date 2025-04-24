import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import os

# Hyperparameters
class Config:
    embedding_dim = 300
    hidden_dim = 512
    num_layers = 2
    dropout = 0.2
    batch_size = 64
    seq_length = 20
    learning_rate = 0.001
    epochs = 5
    min_word_freq = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

class Vocabulary:
    def __init__(self, min_freq=config.min_word_freq):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_count = {}
        self.min_freq = min_freq
        
    def build_vocab(self, text_files):
        word_counter = Counter()
        for file_path in text_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    words = line.strip().split()
                    word_counter.update(words)
        
        idx = len(self.word2idx)
        for word, count in word_counter.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        self.word_count = dict(word_counter)
        print(f"Vocabulary size: {len(self.word2idx)}")
        
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, word):
        return self.word2idx.get(word, self.word2idx['<UNK>'])
    
    def decode(self, idx):
        return self.idx2word.get(idx, '<UNK>')

class TextDataset(Dataset):
    def __init__(self, file_path, vocab, seq_length=config.seq_length):
        self.vocab = vocab
        self.seq_length = seq_length
        self.data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                if len(words) >= 2:  # Need at least 2 words for input and target
                    indices = [vocab.encode(word) for word in words]
                    for i in range(0, len(indices) - self.seq_length):
                        seq = indices[i:i+self.seq_length]
                        target = indices[i+1:i+self.seq_length+1]  # Shifted by 1
                        self.data.append((seq, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq), torch.tensor(target)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.dropout(x)
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        return (h0, c0)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        hidden = model.init_hidden(inputs.size(0), device)
        output, _ = model(inputs, hidden)
        
        # Reshape output and targets for calculation
        output = output.reshape(-1, output.shape[2])
        targets = targets.reshape(-1)
        
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            hidden = model.init_hidden(inputs.size(0), device)
            output, _ = model(inputs, hidden)
            
            output = output.reshape(-1, output.shape[2])
            targets = targets.reshape(-1)
            
            loss = criterion(output, targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def predict_next_word(model, text, vocab, device, top_k=5):
    model.eval()
    words = text.split()[-config.seq_length:]
    
    # Pad the sequence if needed
    if len(words) < config.seq_length:
        words = ['<PAD>'] * (config.seq_length - len(words)) + words
    
    # Convert words to indices
    indices = [vocab.encode(word) for word in words]
    
    # Convert to tensor and add batch dimension
    x = torch.tensor([indices]).to(device)
    
    with torch.no_grad():
        hidden = model.init_hidden(1, device)
        output, _ = model(x, hidden)
        
        # Get the prediction for the next word (last position in sequence)
        predicted = output[0, -1, :]
        
        # Get top-k predictions
        probs, indices = torch.topk(torch.softmax(predicted, dim=0), top_k)
        
        results = []
        for i in range(top_k):
            word = vocab.decode(indices[i].item())
            prob = probs[i].item()
            results.append((word, prob))
        
        return results

def main():
    # Set up paths
    train_path = 'train.txt'
    valid_path = 'valid.txt'
    test_path = 'test.txt'
    
    # Build vocabulary
    vocab = Vocabulary(min_freq=config.min_word_freq)
    vocab.build_vocab([train_path, valid_path, test_path])
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_path, vocab, config.seq_length)
    valid_dataset = TextDataset(valid_path, vocab, config.seq_length)
    test_dataset = TextDataset(test_path, vocab, config.seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # Initialize model
    model = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(config.device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.encode('<PAD>'))
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_valid_loss = float('inf')
    for epoch in range(config.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, config.device)
        valid_loss = evaluate(model, valid_loader, criterion, config.device)
        
        print(f'Epoch: {epoch+1}/{config.epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print('Model saved!')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Test the model
    test_loss = evaluate(model, test_loader, criterion, config.device)
    print(f'Test Loss: {test_loss:.4f}')
    
    # Example predictions
    sample_text = "I want to"
    predictions = predict_next_word(model, sample_text, vocab, config.device)
    print(f"\nInput: '{sample_text}'")
    print("Predicted next words:")
    for word, prob in predictions:
        print(f"  {word}: {prob:.4f}")

if __name__ == "__main__":
    main()