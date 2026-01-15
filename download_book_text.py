import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def download_text(url, timeout=10):
    """
    Download text from a given URL with timeout and error handling.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logging.error(f"Failed to download text from {url}: {e}")
        return ""

def concatenate_novels(novel_urls):
    """
    Concatenate multiple novels downloaded from the given URLs.
    """
    concatenated_text = ""
    for url in novel_urls:
        logging.info(f"Downloading {url}...")
        text = download_text(url)
        if text:
            concatenated_text += text.strip() + "\n\n"
    return concatenated_text

if __name__ == "__main__":
    # List of URLs for public domain novels (Gutenberg)
    novel_urls = [
        "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
        "https://www.gutenberg.org/files/2600/2600-0.txt"   # War and Peace
    ]

    # Download and concatenate novels
    concatenated_novels = concatenate_novels(novel_urls)

    if concatenated_novels:
        with open("data.txt", "w", encoding="utf-8") as file:
            file.write(concatenated_novels)
        logging.info("Text dataset saved to data.txt")
    else:
        logging.error("No text was downloaded.")
