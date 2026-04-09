# DOC 3 — Session Log

**v12 · Claude's memory across sessions**

_Claude fills this at the end of every session. The human saves it and uploads it at the start of every future session. This document is the only continuity between chats._

Keep the last 5 session entries. Move critical decisions to the Permanent Decisions section at the bottom. Delete older entries but never delete the Permanent Decisions.

---

## Project snapshot

| Field               | Value                                                                                                                                                                                                                         |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Project name        | My LLM from scratch                                                                                                                                                                                                           |
| Current phase       | Phase 4 — Transformer architecture (in progress)                                                                                                                                                                              |
| Overall progress    | 80% of Phase 4 — transformer.py written and verified on GPU. One concept remains before phase is complete: causal masking (LEARNING session). Phase 5 begins after that. |
| Last session date   | 9 April 2026                                                                                                                                                                                                                  |
| Last session type   | BUILD                                                                                                                                                                                                                         |
| Next session type   | LEARNING                                                                                                                                                                                                                      |
| Next immediate task | Causal masking — what it is, what the mask matrix looks like, and why the model produces garbage without it. One focused concept, one session.                                                                                |

---

## Phase and concept progress

| Phase                              | Status      | Notes                                                                                                                                                                                                                                                                          |
| ---------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Phase 1 — Python + math            | COMPLETE    | All core concepts understood. NumPy practised in code. Chain rule, functions, loops, classes done.                                                                                                                                                                             |
| Phase 2 — PyTorch fundamentals     | COMPLETE    | Tensors, autograd, forward pass, MSE loss, backward pass, training loop all covered and practised in code. Matplotlib plots added. Optimizer (SGD/Adam) not formally covered — introduced naturally in Phase 3.                                                                |
| Phase 3 — First neural network     | COMPLETE    | Neuron, nn.Module, nn.Linear, Adam, two-layer network, multiple input features, hidden layer representations, convergence, Softmax, cross-entropy loss, classification all understood. Two permanent decisions made: no Matplotlib from Session 8, GPU enabled from Session 8. |
| Phase 4 — Transformer architecture | IN PROGRESS | All concepts understood. transformer.py written and verified — 816,512 parameters, forward pass confirmed on GPU. Causal masking (LEARNING) remains before phase is complete.                                                                                                 |
| Phase 5 — Dataset + pipeline       | NOT STARTED |                                                                                                                                                                                                                                                                                |
| Phase 6 — Train the model          | NOT STARTED |                                                                                                                                                                                                                                                                                |
| Phase 7 — Serve it locally         | NOT STARTED |                                                                                                                                                                                                                                                                                |

---

## Session entries

Most recent at the top. Keep last 5 entries only.

---

### Session 11 — BUILD: transformer.py

| Field              | Value        |
| ------------------ | ------------ |
| Date               | 9 April 2026 |
| Session type       | BUILD        |
| Mode stayed clean? | Yes          |

**What was accomplished**

- Wrote `model/transformer.py` — the full decoder-only Transformer model
- Four components: `make_causal_mask`, `MultiHeadSelfAttention`, `FeedForward`, `TransformerBlock`, `TransformerLM`
- Hyperparameters set: d_model=128, n_heads=4, n_layers=4, max_seq_len=128, ffn_dim=512, dropout=0.1
- Weight tying implemented — output head shares weights with token embedding table
- Forward pass verified on GPU: input `(2, 32)` → output `(2, 32, 65)` ✓
- Total parameters: 816,512 (lower than estimated ~1.05M because weight tying means the output head and embedding table share weights rather than being counted twice — correct behaviour)

**What is unfinished or unclear**

- Causal masking introduced in code comments but not yet covered as a focused Learning concept — next session
- `vocab_size` is placeholder (65) — real value set in Phase 5 when dataset is chosen

**Exact next task**

LEARNING session: causal masking — what it is, what the mask matrix looks like, and what the model would predict incorrectly without it.

---

### Session 10 — Self-attention, multi-head attention, feed-forward layer, layer norm, residual connections

| Field              | Value        |
| ------------------ | ------------ |
| Date               | 9 April 2026 |
| Session type       | LEARNING     |
| Mode stayed clean? | Yes          |

**What was accomplished**

- Introduced self-attention — the mechanism by which every token attends to every other token via learned Query, Key, Value vectors
- Library analogy for Q/K/V: Query = what you search for, Key = label on the spine, Value = content inside. All three roles held simultaneously by every token.
- Explained the five steps of scaled dot-product attention: create Q/K/V → dot product scores → scale by √d_k → Softmax → weighted sum of Values
- Warm-up 1: human correctly derived attention scores shape `(6,6)` for 6-token sequence, and unprompted identified each row as a Query vector — not just "one per token"
- Introduced multi-head attention — h heads run in parallel, each with own W_Q/W_K/W_V, outputs concatenated then projected by W_O back to `(seq_len, d_model)`
- Warm-up 2: human correctly computed d_k = 16 (d_model=64 ÷ h=4) and derived concatenated shape `(6, 64)` before W_O projection
- Introduced feed-forward layer — two linear layers with GELU between them. Expands to 4×d_model then compresses back. Processes each token independently.
- Warm-up 3: human correctly derived FFN shapes `(5, 256)` and `(5, 64)` without hesitation. No "ig" needed — clean derivation.
- Introduced GELU — smoother ReLU, standard activation for Transformers
- Introduced layer normalisation — centres each token's vector, keeps activations stable, shape in = shape out
- Introduced residual connections — `x = x + block(x)`. Human initially deflected ("you say") — pushed back, human then correctly reasoned that addition requires matching shapes. Good correction moment.
- Introduced Pre-LN ordering — LayerNorm before attention and FFN, not after
- Completed full Transformer block: `x = x + Attention(LayerNorm(x))` then `x = x + FFN(LayerNorm(x))`. Human can read and explain the full block diagram.

**Concepts covered**

- Self-attention — UNDERSTOOD
- Query / Key / Value — UNDERSTOOD
- Attention scores matrix — CAN EXPLAIN
- Scaled dot-product attention (√d_k) — INTRODUCED
- Multi-head attention — UNDERSTOOD
- d_k = d_model ÷ h — CAN EXPLAIN
- Feed-forward layer — CAN EXPLAIN
- GELU activation — INTRODUCED
- Layer normalisation — UNDERSTOOD
- Residual connection — UNDERSTOOD
- Pre-LN — UNDERSTOOD
- Full Transformer block structure — CAN EXPLAIN

**What is unfinished or unclear**

- Causal masking not yet covered — next Learning session before Phase 5

**Exact next task**

BUILD session: write `model/transformer.py` — the full Transformer model in code.

---

### Session 9 — Tokenisation, embeddings, positional encoding

| Field              | Value        |
| ------------------ | ------------ |
| Date               | 9 April 2026 |
| Session type       | LEARNING     |
| Mode stayed clean? | Yes          |

**What was accomplished**

- Introduced tokenisation — splitting text into units the model can process. Character-level chosen for this project.
- Introduced token IDs — integers that label tokens. "Locker numbers, not measurements."
- Introduced embeddings — a lookup table that converts token IDs into vectors. Shape: `(vocab_size, d_model)`. Every row is a learned vector.
- Human correctly reasoned that the word "cat" appearing in different contexts needs the same base vector before context is added — embedding table returns the same row each time.
- Introduced positional encoding — a second embedding table, shape `(max_seq_len, d_model)`, indexed by position rather than token identity. Added element-wise to token embeddings.
- Human correctly noted that position 0 always returns the same positional vector regardless of which token is there — confirmed as correct behaviour.
- Decided on learned positional embeddings (not sinusoidal) — GPT-style, simpler to implement.

**Concepts covered**

- Token / Token ID — UNDERSTOOD
- Embedding / Embedding table — UNDERSTOOD
- Positional encoding (learned) — UNDERSTOOD
- d_model — UNDERSTOOD
- Sequence length — UNDERSTOOD

**What is unfinished or unclear**

- Sinusoidal positional encoding ruled out — not needed for this project

**Exact next task**

LEARNING session: self-attention — Q/K/V, scaled dot-product attention, multi-head attention, feed-forward layer, layer norm, residual connections.

---

### Session 8 — Softmax, cross-entropy, classification network

| Field              | Value         |
| ------------------ | ------------- |
| Date               | 29 March 2026 |
| Session type       | LEARNING      |
| Mode stayed clean? | Yes           |

**What was accomplished**

- Introduced Softmax — turns raw scores into probabilities. All outputs positive, all sum to 1.0.
- Introduced cross-entropy loss — `-log(probability of correct class)`. Punishes confident wrong answers hardest.
- Introduced logits — raw output scores before Softmax is applied
- Wrote `network3.py` — classification network. 9 examples, 3 classes.
- Human correctly identified `nn.CrossEntropyLoss` applies Softmax internally — do not apply it yourself first
- GPU enabled from this session onward — standard `.to(device)` pattern established
- No Matplotlib from this session onward — human prefers print-only

**Concepts covered**

- Logits — UNDERSTOOD
- Softmax — UNDERSTOOD
- Cross-entropy loss — UNDERSTOOD
- `nn.CrossEntropyLoss` — UNDERSTOOD
- `torch.argmax` — UNDERSTOOD
- Accuracy — UNDERSTOOD

**What is unfinished or unclear**

- None — Phase 3 complete

**Exact next task**

LEARNING session: tokenisation and embeddings — how text becomes numbers the Transformer can process.

---

### Session 7 — Multiple input features

| Field              | Value         |
| ------------------ | ------------- |
| Date               | 29 March 2026 |
| Session type       | LEARNING      |
| Mode stayed clean? | Yes           |

**What was accomplished**

- Extended `network.py` to two inputs — wrote `network2.py`
- Network correctly discovered y = 2x₁ + 3x₂ from 200 training examples
- Human resolved key misconception: inputs are given, weights are learned
- Correctly computed warm-up: z = (2×3) + (−1×4) + 0.5 = 2.5, ReLU → 2.5
- Noted GPU not used — explained PyTorch defaults to CPU; will add `.to("cuda")` properly from Session 8

**Concepts covered**

- Multiple input features / weighted sum — UNDERSTOOD
- Distributed representations — UNDERSTOOD
- Convergence (empirically discovered) — UNDERSTOOD

**What is unfinished or unclear**

- GPU not yet active — deferred to Session 8

**Exact next task**

LEARNING session: Softmax and cross-entropy loss. What happens when the output is a category instead of a number. Build a small classification network. Phase 3 complete after this.

---

## Permanent decisions archive

| Decision                                       | Made in session   | Reason — never change because...                                                                                                                                                                                                                                                       |
| ---------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Build on local PC only — no cloud              | Session 0 (setup) | The goal is to understand every part of the system. Cloud abstracts too much.                                                                                                                                                                                                          |
| Language: Python + PyTorch                     | Session 0 (setup) | Industry standard. Best learning resources. GPU support built in.                                                                                                                                                                                                                      |
| Use Python 3.11 venv, not system Python 3.14   | Session 1 (setup) | PyTorch has no CUDA build for Python 3.14. Always activate venv before any work. Command: C:\projects\myLLM\venv\Scripts\activate                                                                                                                                                      |
| Documents stored as .md files, not .docx       | Session 1 (setup) | Markdown is plain text — easier to read, edit, version, and paste into Claude.                                                                                                                                                                                                         |
| Character-level tokenizer                      | Session 1         | Simplest to implement, easiest to understand. Will upgrade to BPE in Phase 5 when the full pipeline is built.                                                                                                                                                                          |
| Use VS Code play button to run files           | Session 3         | Ensures the correct venv Python is used. Running python directly in terminal can pick up wrong interpreter.                                                                                                                                                                            |
| Human copies code, does not type it            | Session 5         | Comments are half the learning. Human confirmed they would skip comments if writing themselves. Copying is faster and preserves the full annotated explanation.                                                                                                                        |
| No Matplotlib from Session 8 onward            | Session 8         | Human prefers print-only output — faster to run, less distraction. All loss tracking via print statements only.                                                                                                                                                                        |
| GPU enabled from Session 8 onward              | Session 8         | Phase 3 complete — all learning concepts done. Standard pattern every file must use: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, then `.to(device)` on inputs, labels, and model. Data and model must always be on the same device or PyTorch will error. |
| Learned positional embeddings (not sinusoidal) | Session 9         | GPT-style architecture. Same `nn.Embedding` as token embeddings but indexed by position. Simpler to implement and understand. Matches the reference architecture. Sinusoidal encoding is not needed for this project.                                                                  |
| GELU activation function                       | Session 10        | Standard for GPT-style Transformers. Smoother than ReLU — better gradient flow. Used in the feed-forward layer of every Transformer block.                                                                                                                                             |
| Pre-LN (layer norm before sub-blocks)          | Session 10        | Modern standard. LayerNorm applied before attention and before FFN. More stable during training than Post-LN.                                                                                                                                                                          |
| Residual connections throughout                | Session 10        | `x = x + block(x)` after every attention and FFN sub-block. Required for training stability. Non-negotiable in any Transformer.                                                                                                                                                        |
| Weight tying                                   | Session 11        | Output head shares weights with token embedding table. Standard GPT technique — reduces parameter count and is semantically consistent. Non-negotiable for this architecture.                                                                                                           |

---

_End of Document 3 — Session Log — v12_
