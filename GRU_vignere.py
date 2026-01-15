import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
import string

def vigenere_cipher(text, key):
    alphabet = string.ascii_lowercase
    key = key.lower()
    key_index = 0
    encrypted_text = ""
    for char in text:
        if char.isalpha()and char.lower() in alphabet:
            if char.isupper():
                shift = alphabet.index(key[key_index % len(key)]) - 26
            else:
                shift = alphabet.index(key[key_index % len(key)])
            shifted_alphabet = alphabet[shift:] + alphabet[:shift]
            if char.isupper():
                shifted_alphabet = shifted_alphabet.upper()
            elif char.islower():
                shifted_alphabet = shifted_alphabet.lower()
            encrypted_char = shifted_alphabet[alphabet.index(char.lower())]
            encrypted_text += encrypted_char
            key_index += 1
        else:
            encrypted_text += char  # Add non-alphabetic characters directly
    return encrypted_text

def vigenere_decrypt(text, key):
    decrypted_text = ""
    alphabet = string.ascii_lowercase
    key = key.lower()
    key_index = 0
    for char in text:
        if char.isalpha() and char.lower() in alphabet:
            if char.isupper():
                shift = alphabet.index(key[key_index % len(key)]) - 26
            else:
                shift = alphabet.index(key[key_index % len(key)])
            shifted_alphabet = alphabet[shift:] + alphabet[:shift]
            if char.isupper():
                shifted_alphabet = shifted_alphabet.upper()
            elif char.islower():
                shifted_alphabet = shifted_alphabet.lower()
            decrypted_char = alphabet[shifted_alphabet.index(char.lower())]
            decrypted_text += decrypted_char
            key_index += 1
        else:
            decrypted_text += char  # Add non-alphabetic characters directly
    return decrypted_text

# Load the original text from 'alice.txt'
with open('alice.txt', 'r') as file:
    original_text = file.read()

key = "tired"

# Encrypt the original text using Caesar cipher
encrypted_text = vigenere_cipher(original_text, key)
sample_text = "this is a key langss magnificient"
print(f"Original Text: {sample_text}")
print(f"Encrypted Text: {encrypted_text}")

# Write the encrypted text to 'encrypted.txt'
with open('vigenere_encrypted.txt', 'w') as file:
    file.write(encrypted_text)


# Load the text dataset
text_field = Field(lower=True, tokenize=list)
train_dataset = TranslationDataset(
    path='', exts=('vigenere_encrypted.txt', 'alice.txt'),
    fields=(('encrypted', text_field), ('decrypted', text_field)))

text_field.build_vocab(train_dataset)

# Define the GRU model
class GRUDecryptor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(GRUDecryptor, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, _ = self.gru(embedded)
        output = self.fc(output.view(-1, self.hidden_dim))
        return output.view(input.shape[0], -1, output.shape[-1])

# Define training parameters
INPUT_DIM = len(text_field.vocab)
HIDDEN_DIM = 256
OUTPUT_DIM = len(text_field.vocab)
N_LAYERS = 2
DROPOUT = 0.15
BATCH_SIZE = 32
N_EPOCHS = 30

print(INPUT_DIM)
print(OUTPUT_DIM)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
model = GRUDecryptor(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).cuda()
optimizer = optim.Adam(model.parameters())

# Training loop
train_iterator, _ = BucketIterator.splits(
    (train_dataset, train_dataset), batch_size=BATCH_SIZE,
    sort_within_batch=True, sort_key=lambda x: len(x.encrypted),
    device=torch.device('cuda'))

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0
    for batch in train_iterator:
        optimizer.zero_grad()
        encrypted_data = batch.encrypted.cuda()
        decrypted_data = batch.decrypted.cuda()
        output = model(encrypted_data)
        loss = criterion(output.view(-1, OUTPUT_DIM), decrypted_data.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f'Epoch: {epoch+1:02} | Loss: {epoch_loss:.4f}')

# Function to decrypt encrypted text
def decrypt_text_gru(encrypted_text):
    encrypted_text = encrypted_text.lower()
    encrypted_text_int = torch.tensor([text_field.vocab.stoi[char] for char in encrypted_text], device=torch.device('cuda'))
    decrypted_text_int = model(encrypted_text_int.unsqueeze(1)).argmax(dim=-1).squeeze().cpu().numpy()
    decrypted_text = ''.join([text_field.vocab.itos[i] for i in decrypted_text_int])
    return decrypted_text

# Encrypt a sample text
sample_text = "hello world my name is alice"
key = "alice"
encrypted_text = vigenere_cipher(sample_text, key)

# Decrypt the encrypted text using the trained model
decrypted_text = decrypt_text_gru(encrypted_text)

print(f"Original Text: {sample_text}")
print(f"Encrypted Text: {encrypted_text}")
print(f"Decrypted Text: {decrypted_text}")
