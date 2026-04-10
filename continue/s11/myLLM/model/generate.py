# =============================================================================
# generate.py
# Phase 6 — Text generation from a trained checkpoint
#
# What this file does:
#   1. Loads the saved model weights from runs/best_model.pt
#   2. Takes a seed string (e.g. "ROMEO") from you
#   3. Repeatedly asks the model "what comes next?" and samples an answer
#   4. Prints the generated text to the terminal
#
# Run with: python generate.py
# Change SEED, MAX_NEW_TOKENS, and TEMPERATURE below to experiment.
# =============================================================================

import os
import sys
import torch

# Make sure Python can find our modules from the project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.dataset      import download_data, build_tokeniser
from model.transformer import TransformerLM

# =============================================================================
# GENERATION SETTINGS — change these to experiment
# =============================================================================

SEED           = "ROMEO"   # The text the model continues from
MAX_NEW_TOKENS = 500       # How many characters to generate after the seed
TEMPERATURE    = 0.8       # Lower = more focused, higher = more creative (try 0.6–1.2)

# =============================================================================
# MODEL SETTINGS — must match train.py exactly
# =============================================================================

VOCAB_SIZE  = 65
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 4
SEQ_LEN     = 128
FFN_DIM     = 512
DROPOUT     = 0.0          # Always 0.0 at inference — we want deterministic behaviour

CHECKPOINT_PATH = os.path.join(ROOT, 'runs', 'best_model.pt')

# =============================================================================
# SETUP
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[generate] Device     : {device}")

# Build the tokeniser from the dataset (we need char_to_id and id_to_char)
download_data()
DATA_PATH = os.path.join(ROOT, 'data', 'input.txt')
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw_text = f.read()

_, vocab_size, char_to_id, id_to_char = build_tokeniser(raw_text)

# =============================================================================
# LOAD MODEL
# =============================================================================

model = TransformerLM(
    vocab_size  = VOCAB_SIZE,
    d_model     = D_MODEL,
    n_heads     = N_HEADS,
    n_layers    = N_LAYERS,
    max_seq_len = SEQ_LEN,
    ffn_dim     = FFN_DIM,
    dropout     = DROPOUT,
)

# Load the saved weights into the model
# map_location=device handles the case where the checkpoint was saved on GPU
# but you are loading on CPU (or vice versa)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()   # Dropout OFF — we want consistent output at inference time

print(f"[generate] Checkpoint loaded from {CHECKPOINT_PATH}")

# =============================================================================
# GENERATION FUNCTION
# =============================================================================

@torch.no_grad()   # No gradient tracking needed — we are not training
def generate(seed_text, max_new_tokens, temperature):
    """
    Generate text by repeatedly predicting the next token.

    Arguments:
      seed_text       : string — the model continues from this
      max_new_tokens  : how many new characters to generate
      temperature     : controls randomness (see top of file)

    How it works:
      1. Encode the seed string into token IDs
      2. Feed the token IDs into the model — get logits for every position
      3. Take only the logits at the LAST position (that's the next-token prediction)
      4. Divide by temperature — this sharpens or flattens the distribution
      5. Apply Softmax to get probabilities
      6. Sample one token ID from that distribution
      7. Append it to the sequence and repeat from step 2

    This is called autoregressive generation — the model's own output
    becomes its next input.
    """

    # Encode the seed — produces a list of integer token IDs
    ids = [char_to_id[ch] for ch in seed_text if ch in char_to_id]

    # Convert to a (1, T) tensor — batch size of 1
    context = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    generated = seed_text   # We'll build up the full output string here

    for _ in range(max_new_tokens):

        # If the context is longer than max_seq_len, trim to the last seq_len tokens.
        # The model cannot handle sequences longer than it was trained on.
        context_trimmed = context[:, -SEQ_LEN:]

        # Forward pass — logits shape: (1, T, vocab_size)
        logits = model(context_trimmed)

        # Take only the last position's logits — this is the next-token prediction
        # Shape: (1, vocab_size) → squeeze to (vocab_size,)
        next_logits = logits[0, -1, :]   # shape: (vocab_size,)

        # Apply temperature — dividing by < 1.0 makes the distribution sharper
        # (high-probability tokens become even more likely)
        next_logits = next_logits / temperature

        # Convert logits to probabilities
        probs = torch.softmax(next_logits, dim=-1)

        # Sample one token ID from the probability distribution
        # torch.multinomial draws one sample according to the probabilities
        next_id = torch.multinomial(probs, num_samples=1).item()

        # Decode the token ID back to a character and add it to our output
        next_char = id_to_char[next_id]
        generated += next_char

        # Append the new token ID to the context for the next iteration
        next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        context = torch.cat([context, next_tensor], dim=1)

    return generated

# =============================================================================
# RUN
# =============================================================================

print(f"\n[generate] Seed       : {repr(SEED)}")
print(f"[generate] Tokens     : {MAX_NEW_TOKENS}")
print(f"[generate] Temperature: {TEMPERATURE}")
print(f"\n{'─' * 60}\n")

output = generate(SEED, MAX_NEW_TOKENS, TEMPERATURE)
print(output)

print(f"\n{'─' * 60}")
print("[generate] Done.")