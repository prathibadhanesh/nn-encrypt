import torch
import string

def get_device():
    """
    Returns the best available device (CUDA or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def caesar_cipher(text, shift, alphabet=string.ascii_lowercase):
    """
    Encrypts/decrypts text using Caesar cipher. 
    Handles both lowercase and uppercase while preserving case and non-alphabetic characters.
    """
    result = []
    for char in text:
        if char.isalpha():
            is_upper = char.isupper()
            char_lower = char.lower()
            if char_lower in alphabet:
                idx = alphabet.index(char_lower)
                new_idx = (idx + shift) % len(alphabet)
                new_char = alphabet[new_idx]
                result.append(new_char.upper() if is_upper else new_char)
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)

def vigenere_cipher(text, key, encrypt=True, alphabet=string.ascii_lowercase):
    """
    Encrypts/decrypts text using Vigenère cipher.
    Handles both lowercase and uppercase while preserving case and non-alphabetic characters.
    """
    result = []
    key = key.lower()
    key_idx = 0
    for char in text:
        if char.isalpha():
            is_upper = char.isupper()
            char_lower = char.lower()
            if char_lower in alphabet:
                key_char = key[key_idx % len(key)]
                shift = alphabet.index(key_char)
                if not encrypt:
                    shift = -shift
                
                char_idx = alphabet.index(char_lower)
                new_idx = (char_idx + shift) % len(alphabet)
                new_char = alphabet[new_idx]
                result.append(new_char.upper() if is_upper else new_char)
                key_idx += 1
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)

if __name__ == "__main__":
    # Smoke tests
    test_text = "Hello World! 123"
    c_shift = 3
    c_encrypted = caesar_cipher(test_text, c_shift)
    c_decrypted = caesar_cipher(c_encrypted, -c_shift)
    print(f"Caesar: {test_text} -> {c_encrypted} -> {c_decrypted}")
    assert test_text == c_decrypted

    v_key = "secret"
    v_encrypted = vigenere_cipher(test_text, v_key, encrypt=True)
    v_decrypted = vigenere_cipher(v_encrypted, v_key, encrypt=False)
    print(f"Vigenere: {test_text} -> {v_encrypted} -> {v_decrypted}")
    assert test_text == v_decrypted
    print("All smoke tests passed!")
