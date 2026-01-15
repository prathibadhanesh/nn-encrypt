import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import string
import time
import logging
from utils import caesar_cipher, get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class CharTokenizer:
    def __init__(self, chars=None):
        if chars is None:
            # Basic set of characters
            chars = sorted(list(set(string.printable)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi.get(c, self.stoi.get(' ', 0)) for c in text]

    def decode(self, tokens):
        return "".join([self.itos.get(i, '?') for i in tokens.tolist()])

class CipherDataset(Dataset):
    def __init__(self, text, shift, seq_len=100):
        self.tokenizer = CharTokenizer()
        self.seq_len = seq_len
        # Ensure we work with lowercase for the target to simplify training
        self.original_text = text.lower()
        self.encrypted_text = caesar_cipher(text, shift).lower()
        
        # Prepare sequences
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
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))
        # embedded: (batch_size, seq_len, hidden_dim)
        output, _ = self.gru(embedded)
        # output: (batch_size, seq_len, hidden_dim)
        logits = self.fc(output)
        # logits: (batch_size, seq_len, output_dim)
        return logits

def train():
    device = get_device()
    logging.info(f"Using device: {device}")

    # Load data
    try:
        with open('alice.txt', 'r') as file:
            original_text = file.read()
    except FileNotFoundError:
        logging.error("alice.txt not found. Please run download_book_text.py first.")
        return

    shift = 6
    dataset = CipherDataset(original_text, shift)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model init
    input_dim = dataset.tokenizer.vocab_size
    hidden_dim = 256
    output_dim = input_dim
    n_layers = 2
    dropout = 0.15
    
    model = GRUDecryptor(input_dim, hidden_dim, output_dim, n_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    n_epochs = 15
    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            # output: (batch, seq, vocab), y: (batch, seq)
            loss = criterion(output.view(-1, output_dim), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output.view(-1, output_dim), y.view(-1))
                val_loss += loss.item()

        logging.info(f'Epoch: {epoch+1:02} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}')

    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")

    # Final Demo
    model.eval()
    try:
        with open('birds.txt', 'r') as file:
            sample_text = file.read()
    except FileNotFoundError:
        sample_text = "This is a fallback sample text for testing."

    encrypted_sample = caesar_cipher(sample_text, shift).lower()
    tokens = torch.tensor(dataset.tokenizer.encode(encrypted_sample), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tokens)
        preds = output.argmax(dim=-1).squeeze(0)
    
    decrypted_sample = dataset.tokenizer.decode(preds)

    print("\n" + "="*50)
    print(f"Original Text: {sample_text[:100]}...")
    print(f"Encrypted Text: {encrypted_sample[:100]}...")
    print(f"Decrypted Text: {decrypted_sample[:100]}...")
    print("="*50 + "\n")

if __name__ == "__main__":
    train()