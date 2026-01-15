import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import string
import time
import logging
from utils import vigenere_cipher, get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class CharTokenizer:
    def __init__(self, chars=None):
        if chars is None:
            chars = sorted(list(set(string.printable)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi.get(c, self.stoi.get(' ', 0)) for c in text]

    def decode(self, tokens):
        return "".join([self.itos.get(i, '?') for i in tokens.tolist()])

class CipherDataset(Dataset):
    def __init__(self, text, key, seq_len=100):
        self.tokenizer = CharTokenizer()
        self.seq_len = seq_len
        self.original_text = text.lower()
        self.encrypted_text = vigenere_cipher(text, key, encrypt=True).lower()
        
        self.enc_tokens = self.tokenizer.encode(self.encrypted_text)
        self.dec_tokens = self.tokenizer.encode(self.original_text)
        
        self.num_samples = len(self.enc_tokens) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = torch.tensor(self.enc_tokens[start:end], dtype=torch.long)
        y = torch.tensor(self.dec_tokens[start:end], dtype=torch.long)
        return x, y

class GRUDecryptor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(GRUDecryptor, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, _ = self.gru(embedded)
        logits = self.fc(output)
        return logits

def train():
    device = get_device()
    logging.info(f"Using device: {device}")

    # Load data
    try:
        with open('alice.txt', 'r') as file:
            original_text = file.read()
    except FileNotFoundError:
        logging.error("alice.txt not found.")
        return

    key = "tired"
    dataset = CipherDataset(original_text, key)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = GRUDecryptor(dataset.tokenizer.vocab_size, 256, dataset.tokenizer.vocab_size, 2, 0.15).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    n_epochs = 30
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, dataset.tokenizer.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output.view(-1, dataset.tokenizer.vocab_size), y.view(-1))
                val_loss += loss.item()

        logging.info(f'Epoch: {epoch+1:02} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}')

    # Save model (optional but good practice)
    torch.save(model.state_dict(), "vigenere_decryptor.pth")

    # Demo
    sample_text = "hello world my name is alice"
    encrypted_sample = vigenere_cipher(sample_text, key).lower()
    tokens = torch.tensor(dataset.tokenizer.encode(encrypted_sample), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tokens)
        preds = output.argmax(dim=-1).squeeze(0)
    
    decrypted_sample = dataset.tokenizer.decode(preds)

    print("\n" + "="*50)
    print(f"Original Text: {sample_text}")
    print(f"Encrypted Text: {encrypted_sample}")
    print(f"Decrypted Text: {decrypted_sample}")
    print("="*50 + "\n")

if __name__ == "__main__":
    train()
