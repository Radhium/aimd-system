# =============================================================================
# train.py
# Phase 7 — Experiment #4: scaled-up model, fixed learning rate
#
# This file ties everything together:
#   - Loads and tokenises the dataset  (dataset.py)
#   - Builds the Transformer model     (model/transformer.py)
#   - Runs the training loop
#   - Prints train + val loss every EVAL_INTERVAL steps
#   - Saves a checkpoint whenever validation loss improves
#
# Changes from Experiment #2 (the fixed-LR baseline):
#   D_MODEL  : 128 → 256
#   N_HEADS  :   4 → 8
#   N_LAYERS :   4 → 6
#   FFN_DIM  : 512 → 1024
#   Parameters: ~816K → ~3.2M
#   No cosine schedule — fixed LR only (performed better on this dataset).
#
# If you get a CUDA out-of-memory error, reduce BATCH_SIZE from 32 to 16.
#
# Run with the VS Code play button (venv must be active).
# =============================================================================

import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.dataset       import download_data, build_tokeniser, load_and_split, get_batch
from model.transformer  import TransformerLM

# =============================================================================
# HYPERPARAMETERS — change only these, never scatter magic numbers in the code
# =============================================================================

# --- Data ---
BATCH_SIZE     = 32       # reduce to 16 if CUDA out-of-memory error occurs
SEQ_LEN        = 128      # must match transformer.py max_seq_len

# --- Model (must match transformer.py exactly) ---
VOCAB_SIZE     = 65       # confirmed by dataset.py self-test
D_MODEL        = 256      # was 128
N_HEADS        = 8        # was 4 — d_k stays 32 per head (256 ÷ 8)
N_LAYERS       = 6        # was 4
FFN_DIM        = 1024     # was 512 — always 4 × D_MODEL
DROPOUT        = 0.1

# --- Training ---
MAX_STEPS      = 50000     # same as Experiment #2 — one variable changes at a time
LEARNING_RATE  = 3e-4     # same fixed LR as Experiment #2 — no cosine schedule
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

download_data()

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

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

@torch.no_grad()
def estimate_loss():
    """
    Average loss over EVAL_STEPS random batches from train and val sets.
    Returns a dict: {'train': float, 'val': float}
    """
    losses = {}

    for split_name, split_data in [('train', train_data), ('val', val_data)]:
        model.eval()
        split_losses = []

        for _ in range(EVAL_STEPS):
            x, y = get_batch(split_data, BATCH_SIZE, SEQ_LEN, device)
            logits = model(x)

            B, T, C = logits.shape
            logits_flat  = logits.view(B * T, C)
            targets_flat = y.view(B * T)

            loss = criterion(logits_flat, targets_flat)
            split_losses.append(loss.item())

        losses[split_name] = sum(split_losses) / len(split_losses)
        model.train()

    return losses

# =============================================================================
# TRAINING LOOP
# =============================================================================

os.makedirs(RUNS_DIR, exist_ok=True)

best_val_loss = float('inf')

print(f"\n[train] Starting training — {MAX_STEPS} steps")
print(f"[train] Eval every {EVAL_INTERVAL} steps\n")

model.train()

for step in range(MAX_STEPS):

    if step % EVAL_INTERVAL == 0:
        losses     = estimate_loss()
        train_loss = losses['train']
        val_loss   = losses['val']

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
    x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, device)

    logits = model(x)

    B, T, C      = logits.shape
    logits_flat  = logits.view(B * T, C)
    targets_flat = y.view(B * T)
    loss         = criterion(logits_flat, targets_flat)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# --- Final evaluation ---
losses = estimate_loss()
print(
    f"\n[train] Training complete."
    f"\n[train] Final train loss : {losses['train']:.4f}"
    f"\n[train] Final val loss   : {losses['val']:.4f}"
    f"\n[train] Best checkpoint  : {CHECKPOINT_PATH}"
)