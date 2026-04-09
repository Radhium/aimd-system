"""
transformer.py — The full Transformer model (decoder-only, GPT-style)

This file defines the complete language model architecture.
It is made of three things stacked in order:

  1. Embedding layer     — converts token IDs to vectors
  2. N × Transformer blocks — the core of the model
  3. Output head         — converts final vectors back to vocabulary scores (logits)

Every concept in this file was covered during the Learning sessions.
Read the comments — they explain WHY each line exists, not just what it does.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# These are the knobs that control the size and behaviour of the model.
# They are set here as defaults. You can override them when creating the model.
# ─────────────────────────────────────────────────────────────────────────────

D_MODEL     = 128   # Size of every token's vector throughout the model
N_HEADS     = 4     # Number of attention heads — must divide evenly into D_MODEL
N_LAYERS    = 4     # How many Transformer blocks to stack
MAX_SEQ_LEN = 128   # Maximum number of tokens the model can see at once
FFN_DIM     = 512   # Size of the hidden layer inside the feed-forward network (4 × D_MODEL)
DROPOUT     = 0.1   # Fraction of activations randomly zeroed during training (regularisation)

# D_K = D_MODEL ÷ N_HEADS = 128 ÷ 4 = 32
# This is the dimension of each attention head's Q, K, and V vectors.


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL MASK
# In a decoder-only model, each token is only allowed to attend to tokens
# that came BEFORE it (and itself). It cannot see the future.
# This is enforced with a mask — an upper-triangular matrix of -infinity values.
# When added to the attention scores before Softmax, -infinity → probability 0.
# ─────────────────────────────────────────────────────────────────────────────

def make_causal_mask(seq_len, device):
    """
    Returns a (seq_len, seq_len) causal mask.
    Upper triangle is -inf (future tokens blocked).
    Lower triangle + diagonal is 0 (allowed).

    Example for seq_len = 4:
      [[  0, -inf, -inf, -inf],
       [  0,    0, -inf, -inf],
       [  0,    0,    0, -inf],
       [  0,    0,    0,    0]]
    Token 0 can only see itself.
    Token 3 can see tokens 0, 1, 2, and itself.
    """
    # torch.ones creates a matrix of 1s, triu keeps only the upper triangle
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    # Replace 1s with -inf, 0s stay as 0 — these get added to attention scores
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-HEAD SELF-ATTENTION
# Each token creates a Query (what it's looking for), a Key (what it contains),
# and a Value (what it contributes). Every token attends to every other token,
# but the causal mask blocks future positions.
# Multiple heads run in parallel, each learning different relationships.
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        # Sanity check — d_model must divide evenly into n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Size of each head's Q, K, V vectors

        # Three linear layers that project token embeddings into Q, K, V spaces.
        # Each maps (d_model) → (d_model). We split into heads afterward.
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Output projection — maps concatenated head outputs back to d_model
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        B, T, C = x.shape  # batch size, sequence length, d_model

        # Step 1 — Create Q, K, V for all heads at once
        # Shape after linear: (B, T, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Step 2 — Split into individual heads
        # Reshape from (B, T, d_model) → (B, T, n_heads, d_k)
        # Then transpose to (B, n_heads, T, d_k) so each head sees its own slice
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3 — Compute attention scores
        # Q @ K^T gives similarity between every pair of tokens
        # Shape: (B, n_heads, T, T) — one score per token pair, per head
        scores = Q @ K.transpose(-2, -1)

        # Step 4 — Scale by √d_k to prevent exploding dot products
        # (covered in Session 10 — large dot products kill Softmax gradients)
        scores = scores / (self.d_k ** 0.5)

        # Step 5 — Apply causal mask (block future tokens)
        if mask is not None:
            scores = scores + mask  # -inf positions → probability 0 after Softmax

        # Step 6 — Softmax turns scores into attention weights (sum to 1 per row)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Step 7 — Weighted sum of Values
        # Shape: (B, n_heads, T, d_k)
        out = weights @ V

        # Step 8 — Concatenate heads back together
        # Transpose → (B, T, n_heads, d_k) then reshape → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Step 9 — Final output projection (W_O)
        out = self.W_O(out)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# FEED-FORWARD NETWORK (FFN)
# After attention, each token's vector is processed independently by a small
# two-layer network. This is where the model "thinks" about each token in
# isolation, after it has gathered context from the attention step.
# Expands to 4 × d_model then compresses back — GELU activation in between.
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):

    def __init__(self, d_model, ffn_dim, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),   # Expand: d_model → ffn_dim (4 × d_model)
            nn.GELU(),                      # Smooth activation (not ReLU — covered Session 10)
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),   # Compress back: ffn_dim → d_model
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER BLOCK
# One full Transformer block = attention + FFN, each wrapped with:
#   - Pre-LN (LayerNorm applied BEFORE each sub-block)
#   - Residual connection (x = x + block(x)) applied AFTER each sub-block
#
# Full structure (Pre-LN style):
#   x = x + Attention( LayerNorm(x) )
#   x = x + FFN( LayerNorm(x) )
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)              # LayerNorm before attention
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)              # LayerNorm before FFN
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN attention with residual connection
        x = x + self.dropout(self.attention(self.ln1(x), mask=mask))

        # Pre-LN feed-forward with residual connection
        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x


# ─────────────────────────────────────────────────────────────────────────────
# THE FULL MODEL
# Puts everything together in order:
#   1. Token embedding      — (token ID) → (vector of size d_model)
#   2. Positional embedding — (position index) → (vector of size d_model)
#                             Added element-wise to token embeddings
#   3. N × Transformer blocks
#   4. Final LayerNorm
#   5. Output linear layer  — projects each token vector to (vocab_size) logits
#                             These are the raw scores for each possible next token
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model     = D_MODEL,
        n_heads     = N_HEADS,
        n_layers    = N_LAYERS,
        max_seq_len = MAX_SEQ_LEN,
        ffn_dim     = FFN_DIM,
        dropout     = DROPOUT,
    ):
        super().__init__()

        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # Token embedding table — one learnable vector per token in the vocabulary
        # Shape: (vocab_size, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embedding table — one learnable vector per possible position
        # Shape: (max_seq_len, d_model)
        # (Learned positional embeddings — decided Session 9)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Stack of N Transformer blocks — this is where most computation happens
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        # Final LayerNorm — applied after the last block, before the output head
        self.ln_final = nn.LayerNorm(d_model)

        # Output head — projects each token's vector to a score for every vocab token
        # Shape: (d_model) → (vocab_size)
        # These scores are the logits — passed to CrossEntropyLoss during training
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying — the output head shares weights with the token embedding.
        # This is a standard GPT technique: if two tokens are similar in meaning,
        # they should have similar embeddings AND similar output logits.
        # It also reduces total parameter count slightly.
        self.output_head.weight = self.token_embedding.weight

    def forward(self, token_ids):
        """
        token_ids: integer tensor of shape (batch_size, seq_len)
        Each value is a token ID (index into the vocabulary).

        Returns: logits of shape (batch_size, seq_len, vocab_size)
        These are the model's raw predictions for what token comes next
        at each position in the sequence.
        """
        B, T = token_ids.shape

        # Ensure input doesn't exceed max sequence length
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        device = token_ids.device

        # Step 1 — Look up token embeddings
        # Shape: (B, T, d_model)
        tok_emb = self.token_embedding(token_ids)

        # Step 2 — Create position indices [0, 1, 2, ..., T-1] and look up positional embeddings
        # Shape of positions: (T,) → positional embeddings: (T, d_model)
        positions = torch.arange(T, device=device)
        pos_emb = self.pos_embedding(positions)

        # Step 3 — Add token and positional embeddings element-wise
        # pos_emb is (T, d_model) — PyTorch broadcasts this across the batch dimension
        x = self.dropout(tok_emb + pos_emb)

        # Step 4 — Build the causal mask for this sequence length
        mask = make_causal_mask(T, device)

        # Step 5 — Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # Step 6 — Final LayerNorm
        x = self.ln_final(x)

        # Step 7 — Project to vocabulary size → logits
        # Shape: (B, T, vocab_size)
        logits = self.output_head(x)

        return logits

    def count_parameters(self):
        """Returns the total number of learnable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify the model works
# python transformer.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create a model with a small dummy vocab size for testing
    VOCAB_SIZE = 65  # Will be replaced with real vocab size in Phase 5

    model = TransformerLM(vocab_size=VOCAB_SIZE).to(device)

    print(f"Total parameters: {model.count_parameters():,}")

    # Create a fake batch: 2 sequences, each 32 tokens long
    dummy_input = torch.randint(0, VOCAB_SIZE, (2, 32)).to(device)

    # Run a forward pass
    logits = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")   # Expected: (2, 32)
    print(f"Output shape: {logits.shape}")         # Expected: (2, 32, 65)
    print("Forward pass successful.")
