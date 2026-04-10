# =============================================================================
# train.py
# Phase 6 — Training loop
#
# This file ties everything together:
#   - Loads and tokenises the dataset  (dataset.py)
#   - Builds the Transformer model     (model/transformer.py)
#   - Runs the training loop
#   - Prints train + val loss every EVAL_INTERVAL steps
#   - Saves a checkpoint whenever validation loss improves
#
# Run with the VS Code play button (venv must be active).
# Expected output: loss starting around 4.1, falling steadily over time.
# =============================================================================

import os
import sys
import torch

# Make sure Python can find our modules regardless of where the file lives
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.dataset       import download_data, build_tokeniser, load_and_split, get_batch
from model.transformer  import TransformerLM

# =============================================================================
# HYPERPARAMETERS — change only these, never scatter magic numbers in the code
# =============================================================================

# --- Data ---
BATCH_SIZE     = 32       # how many sequences to process in parallel each step
SEQ_LEN        = 128      # must match transformer.py max_seq_len

# --- Model (must match transformer.py exactly) ---
VOCAB_SIZE     = 65       # confirmed by dataset.py self-test
D_MODEL        = 128
N_HEADS        = 4
N_LAYERS       = 4
FFN_DIM        = 512
DROPOUT        = 0.1

# --- Training ---
MAX_STEPS      = 50000     # total number of training steps
LEARNING_RATE  = 3e-4     # standard starting point for Adam on small Transformers
EVAL_INTERVAL  = 500      # print train + val loss every N steps
EVAL_STEPS     = 50       # how many batches to average for each loss estimate

# --- Checkpointing ---
RUNS_DIR       = os.path.join(ROOT, 'runs')
CHECKPOINT_PATH = os.path.join(RUNS_DIR, 'best_model.pt')

# =============================================================================
# SETUP
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[train] Device        : {device}")

# Download dataset if needed, build tokeniser, encode and split
download_data()

import os as _os
DATA_PATH = os.path.join(ROOT, 'data', 'input.txt')
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw_text = f.read()

_, vocab_size, char_to_id, id_to_char = build_tokeniser(raw_text)
assert vocab_size == VOCAB_SIZE, (
    f"vocab_size mismatch: dataset has {vocab_size}, VOCAB_SIZE set to {VOCAB_SIZE}"
)

train_data, val_data = load_and_split(char_to_id)

# =============================================================================
# MODEL
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
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"[train] Parameters    : {total_params:,}")

# =============================================================================
# OPTIMIZER
# =============================================================================

# Adam with a standard learning rate for small Transformers.
# model.parameters() hands Adam every learnable number in the model.
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loss function: cross-entropy between logits and target token IDs.
# We set ignore_index=-1 as a safety net (not used here but good practice).
criterion = torch.nn.CrossEntropyLoss()

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

@torch.no_grad()                 # no gradient tracking — we are not learning here
def estimate_loss():
    """
    Average loss over EVAL_STEPS random batches from train and val sets.

    Returns a dict: {'train': float, 'val': float}

    Why average over multiple batches?
    A single batch is noisy — it might be unusually easy or hard by chance.
    Averaging over 50 batches gives a much more reliable estimate.
    """
    losses = {}

    for split_name, split_data in [('train', train_data), ('val', val_data)]:
        model.eval()             # dropout OFF — deterministic behaviour
        split_losses = []

        for _ in range(EVAL_STEPS):
            x, y = get_batch(split_data, BATCH_SIZE, SEQ_LEN, device)

            # Forward pass — logits shape: (batch_size, seq_len, vocab_size)
            logits = model(x)

            # CrossEntropyLoss expects:
            #   input  : (N, C) where N = total predictions, C = vocab_size
            #   target : (N,)   where N = total predictions
            # So we flatten batch and sequence dimensions together
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)   # (batch*seq, vocab_size)
            targets_flat = y.view(B * T)           # (batch*seq,)

            loss = criterion(logits_flat, targets_flat)
            split_losses.append(loss.item())

        losses[split_name] = sum(split_losses) / len(split_losses)
        model.train()            # dropout back ON after evaluation

    return losses

# =============================================================================
# TRAINING LOOP
# =============================================================================

os.makedirs(RUNS_DIR, exist_ok=True)

best_val_loss = float('inf')    # track the best validation loss seen so far

print(f"\n[train] Starting training — {MAX_STEPS} steps")
print(f"[train] Eval every {EVAL_INTERVAL} steps\n")

model.train()                   # dropout ON — start in training mode

for step in range(MAX_STEPS):

    # --- Evaluate and print at regular intervals ---
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss   = losses['val']

        # Save a checkpoint if this is the best validation loss we have seen
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            saved_marker = '  ← saved'
        else:
            saved_marker = ''

        print(
            f"step {step:>5} / {MAX_STEPS} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f}"
            f"{saved_marker}"
        )

    # --- Training step ---

    # 1. Get a fresh batch
    x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, device)

    # 2. Forward pass
    logits = model(x)                          # (batch_size, seq_len, vocab_size)

    # 3. Compute loss — flatten as above
    B, T, C      = logits.shape
    logits_flat  = logits.view(B * T, C)
    targets_flat = y.view(B * T)
    loss         = criterion(logits_flat, targets_flat)

    # 4. Backward pass — compute gradients
    optimizer.zero_grad()                      # clear gradients from previous step
    loss.backward()                            # compute new gradients

    # 5. Gradient clipping — prevents very large gradient updates destabilising training
    #    Clips all gradients so their total norm does not exceed 1.0
    #    Standard practice for Transformer training
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # 6. Update weights
    optimizer.step()

# --- Final evaluation after training completes ---
losses = estimate_loss()
print(
    f"\n[train] Training complete."
    f"\n[train] Final train loss : {losses['train']:.4f}"
    f"\n[train] Final val loss   : {losses['val']:.4f}"
    f"\n[train] Best checkpoint  : {CHECKPOINT_PATH}"
)
