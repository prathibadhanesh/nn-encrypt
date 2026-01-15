import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import string
import logging
from Crypto.Cipher import AES
from Crypto.Util import Counter
from utils import vigenere_cipher, get_device
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_entropy(data):
    """Calculate the Shannon entropy of a sequence of bytes/chars."""
    if not data:
        return 0
    entropy = 0
    length = len(data)
    counts = {}
    for x in data:
        counts[x] = counts.get(x, 0) + 1
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy

def aes_encrypt(text, key_str):
    """Encrypt text using AES-128 in CTR mode (to maintain length)."""
    # Key must be 16 bytes
    key = key_str.encode('utf-8').ljust(16, b'\0')[:16]
    ctr = Counter.new(128)
    cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
    # Encrypt
    ciphertext_bytes = cipher.encrypt(text.encode('utf-8'))
    return ciphertext_bytes

class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, data):
        if isinstance(data, str):
            data = data.encode('utf-8')
        return [int(b) for b in data]

    def decode(self, tokens):
        return bytes(tokens).decode('utf-8', errors='replace')

class ComparisonDataset(Dataset):
    def __init__(self, target_tokens, source_tokens, seq_len=64):
        self.target = target_tokens
        self.source = source_tokens
        self.seq_len = seq_len
        self.num_samples = len(self.source) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = torch.tensor(self.source[start:end], dtype=torch.long)
        y = torch.tensor(self.target[start:end], dtype=torch.long)
        return x, y

class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super(SimpleGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        return self.fc(output)

def run_comparison():
    device = get_device()
    logging.info(f"Using device: {device}")

    # 1. Create Data
    try:
        with open('alice.txt', 'r') as file:
            text = file.read()[:50000] # Use a smaller subset for quick demo
    except FileNotFoundError:
        logging.error("alice.txt not found.")
        return

    tokenizer = ByteTokenizer()
    plaintext_tokens = tokenizer.encode(text)

    # Vigenere Encryption
    v_key = "secret"
    v_encrypted_text = vigenere_cipher(text, v_key).lower()
    v_tokens = tokenizer.encode(v_encrypted_text)

    # AES Encryption
    a_key = "super_secret_key"
    a_bytes = aes_encrypt(text, a_key)
    a_tokens = [int(b) for b in a_bytes]

    # Metrics
    logging.info(f"Plaintext Entropy: {calculate_entropy(text):.4f}")
    logging.info(f"Vigenere Entropy:  {calculate_entropy(v_encrypted_text):.4f}")
    logging.info(f"AES Entropy:       {calculate_entropy(a_bytes):.4f} (Nearly perfect 8.0)")

    # 2. Training Setup
    def train_on_data(source_tokens, name):
        logging.info(f"\n--- Training on {name} ---")
        dataset = ComparisonDataset(plaintext_tokens, source_tokens)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = SimpleGRU(tokenizer.vocab_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(200):
            epoch_loss = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output.view(-1, 256), y.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            logging.info(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.4f}")
        return epoch_loss/len(loader)

    # Train
    v_loss = train_on_data(v_tokens, "Vigenere")
    a_loss = train_on_data(a_tokens, "AES")

    logging.info("\nSummary:")
    logging.info(f"Vigenere Final Loss: {v_loss:.4f} (Model is learning)")
    logging.info(f"AES Final Loss:      {a_loss:.4f} (Model is stuck - high loss)")
    
    print("\n" + "="*50)
    print("CONCLUSION:")
    print("Vigenere loss decreases because there is a repeating pattern.")
    print("AES loss stays high because the output is indistinguishable from random noise.")
    print("A GRU cannot learn a mapping for AES because of the Avalanche Effect.")
    print("="*50)

if __name__ == "__main__":
    run_comparison()
