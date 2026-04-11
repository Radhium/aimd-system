# =============================================================================
# dataset.py
# Phase 5 — Data pipeline for character-level language modelling
#
# What this file does, in order:
#   1. Downloads Tiny Shakespeare to data/input.txt (skips if already there)
#   2. Builds the character-level tokeniser from the text
#   3. Encodes the entire text as a list of integers
#   4. Splits into train (90%) and validation (10%) sets
#   5. Exposes get_batch() — returns random (x, y) tensor pairs for training
#
# Nothing in here touches the model. This file's only job is: text → tensors.
# =============================================================================

import os
import urllib.request
import torch

# -----------------------------------------------------------------------------
# 1. DOWNLOAD THE DATASET
# -----------------------------------------------------------------------------

# Where the raw text will live on disk
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_PATH = os.path.join(DATA_DIR, 'input.txt')

# The standard Tiny Shakespeare URL (Andrej Karpathy's hosted copy)
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

def download_data():
    """Download Tiny Shakespeare if it isn't already on disk."""
    os.makedirs(DATA_DIR, exist_ok=True)           # create data/ if it doesn't exist

    if os.path.exists(DATA_PATH):
        print(f"[dataset] Data already exists at {DATA_PATH} — skipping download.")
        return

    print(f"[dataset] Downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"[dataset] Saved to {DATA_PATH}")

# -----------------------------------------------------------------------------
# 2. BUILD THE TOKENISER
# -----------------------------------------------------------------------------

def build_tokeniser(text):
    """
    Given a string of text, return:
      - chars       : sorted list of every unique character in the text
      - vocab_size  : how many unique characters there are
      - char_to_id  : dict mapping each character to an integer ID
      - id_to_char  : dict mapping each integer ID back to a character

    This is the entire character-level tokeniser. No external libraries needed.
    The vocabulary is derived entirely from what characters actually appear in
    the dataset — nothing is assumed in advance.
    """
    chars      = sorted(set(text))          # e.g. ['\n', ' ', '!', ..., 'z']
    vocab_size = len(chars)                 # 65 for Tiny Shakespeare

    # Forward mapping: character → integer
    char_to_id = {ch: i for i, ch in enumerate(chars)}

    # Reverse mapping: integer → character  (used for decoding model output)
    id_to_char = {i: ch for i, ch in enumerate(chars)}

    return chars, vocab_size, char_to_id, id_to_char

# -----------------------------------------------------------------------------
# 3 + 4. ENCODE AND SPLIT
# -----------------------------------------------------------------------------

def encode(text, char_to_id):
    """Convert a string into a list of integer token IDs."""
    return [char_to_id[ch] for ch in text]

def decode(ids, id_to_char):
    """Convert a list of integer token IDs back into a string."""
    return ''.join([id_to_char[i] for i in ids])

def load_and_split(char_to_id, split_ratio=0.9):
    """
    Read the text from disk, encode it, and split into train / val tensors.

    split_ratio=0.9 means 90% training data, 10% validation data.

    Returns:
      train_data : 1D LongTensor of token IDs (90% of the dataset)
      val_data   : 1D LongTensor of token IDs (10% of the dataset)

    Why LongTensor? Because token IDs are integers, and nn.Embedding expects
    integer (Long) type — it uses the ID as an index into the embedding table.
    """
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    ids        = encode(text, char_to_id)
    data       = torch.tensor(ids, dtype=torch.long)    # 1D tensor, length ~1M

    n          = int(len(data) * split_ratio)
    train_data = data[:n]                               # first 90%
    val_data   = data[n:]                               # last 10%

    print(f"[dataset] Total tokens  : {len(data):,}")
    print(f"[dataset] Train tokens  : {len(train_data):,}")
    print(f"[dataset] Val tokens    : {len(val_data):,}")

    return train_data, val_data

# -----------------------------------------------------------------------------
# 5. BATCH SAMPLER
# -----------------------------------------------------------------------------

def get_batch(data, batch_size, seq_len, device):
    """
    Sample a random batch of (input, target) pairs from a data tensor.

    Arguments:
      data       : 1D LongTensor — either train_data or val_data
      batch_size : how many sequences to include in the batch (e.g. 32)
      seq_len    : length of each sequence (must match model's max_seq_len = 128)
      device     : 'cuda' or 'cpu' — tensors are moved here before returning

    How it works:
      - Pick `batch_size` random starting positions in the data
      - For each start position i, grab data[i : i+seq_len]   → this is x
      -                             grab data[i+1 : i+seq_len+1] → this is y
      - x and y are offset by one: every position in x predicts the next
        character, which is the corresponding position in y

    Returns:
      x : LongTensor of shape (batch_size, seq_len) — the inputs
      y : LongTensor of shape (batch_size, seq_len) — the targets (x shifted by 1)
    """
    # Random starting positions — must leave room for seq_len+1 characters
    # torch.randint returns a 1D tensor of `batch_size` random integers
    ix = torch.randint(len(data) - seq_len, (batch_size,))

    # Stack individual sequences into batch tensors
    x  = torch.stack([data[i   : i+seq_len  ] for i in ix])
    y  = torch.stack([data[i+1 : i+seq_len+1] for i in ix])

    # Move to the correct device (GPU or CPU)
    x  = x.to(device)
    y  = y.to(device)

    return x, y

# -----------------------------------------------------------------------------
# SELF-TEST — run this file directly to verify everything works
# python dataset.py   (with venv active)
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[dataset] Device: {device}")

    # Step 1: download
    download_data()

    # Step 2: read text and build tokeniser
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    chars, vocab_size, char_to_id, id_to_char = build_tokeniser(raw_text)

    print(f"[dataset] Vocab size    : {vocab_size}")
    print(f"[dataset] Characters    : {''.join(chars)}")

    # Step 3+4: encode and split
    train_data, val_data = load_and_split(char_to_id)

    # Step 5: sample one batch and verify shapes
    BATCH_SIZE = 4
    SEQ_LEN    = 128

    x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, device)

    print(f"\n[dataset] Batch x shape : {x.shape}")   # expect (4, 128)
    print(f"[dataset] Batch y shape : {y.shape}")   # expect (4, 128)

    # Decode the first sequence in the batch so you can see real text
    print(f"\n[dataset] First x (decoded) :\n{decode(x[0].tolist(), id_to_char)}")
    print(f"\n[dataset] First y (decoded) :\n{decode(y[0].tolist(), id_to_char)}")

    # Verify the offset — x[0][0] and y[0][0] should be adjacent characters
    print(f"\n[dataset] x[0][0] = '{id_to_char[x[0][0].item()]}' (token {x[0][0].item()})")
    print(f"[dataset] y[0][0] = '{id_to_char[y[0][0].item()]}' (token {y[0][0].item()})")
    print(f"[dataset] These should be consecutive characters in the original text.")

    print("\n[dataset] All checks passed.")
