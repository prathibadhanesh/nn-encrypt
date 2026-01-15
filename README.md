# NN-Encrypt: Neural Cryptanalysis with GRUs

Exploring the intersection of Deep Learning and Cryptography. This repository demonstrates the ability of Gated Recurrent Units (GRUs) to learn and reverse classical ciphers while analyzing the resistance of modern encryption (AES).

## Project Overview

This project serves as a proof of concept for **Neural Cryptanalysis**. We use GRU-based recurrent neural networks to "learn" how to decrypt text without being given the underlying encryption key or algorithm logic.

### Key Results
*   **Caesar Cipher**: Easily broken in <5 epochs. 100% accuracy on character mapping.
*   **Vigenère Cipher**: Successfully broken by learning periodic shifts. Loss stabilizes as the model synchronizes with the key length.
*   **AES (Advanced Encryption Standard)**: **Unbreakable** by this architecture. The high entropy and "Avalanche Effect" of AES make it look like random noise to the neural network.

## Repository Structure

*   `utils.py`: Shared encryption utilities and device-agnostic (CPU/CUDA) setup.
*   `download_book_text.py`: Robust data acquisition for training (Gutenberg project).
*   `GRU_decryptor_v2.py`: Training script for the Caesar cipher.
*   `GRU_vigenere_v2.py`: Training script for the Vigenère cipher.
*   `aes_vs_vigenere.py`: Comparison script demonstrating why modern cryptography is resistant to neural analysis.

## Installation & Setup

### 1. Prerequisites
*   Python 3.10+
*   NVIDIA GPU (Optimal) or CPU

### 2. Environment Setup
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Usage

First, download the training data:
```bash
python download_book_text.py
```

Then, run one of the training/comparison scripts:
```bash
# Train on Caesar
python GRU_decryptor_v2.py

# Train on Vigenère
python GRU_vigenere_v2.py

# Run the AES comparison demo
python aes_vs_vigenere.py
```

## Why This Matters

While these models aren't meant to replace real cryptographic tools, they highlight:
1.  **Pattern Discovery**: The power of sequence modeling in identifying hidden linear relationships.
2.  **Cryptographic Limits**: Why modern cryptography (like AES) is mathematically superior to classical "pattern-based" ciphers.
3.  **Modern PyTorch**: A clean, scalable implementation using custom `Dataset` and `DataLoader` classes (no deprecated `torchtext` dependencies).

---
*Created for educational purposes.*
