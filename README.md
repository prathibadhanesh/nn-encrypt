# NN-Encrypt: Neural Cryptanalysis with GRUs

Exploring the intersection of Deep Learning and Cryptography. This repository demonstrates the ability of Gated Recurrent Units (GRUs) to learn and reverse classical ciphers while analyzing the resistance of modern encryption (AES).

## Project Overview

This project serves as a proof of concept for **Neural Cryptanalysis**. GRU-based recurrent neural networks are used to "learn" decryption of text without prior knowledge of the underlying encryption key or algorithm logic.

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

---
