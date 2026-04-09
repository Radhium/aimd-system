# DOC 3 — Session Log

**v11 · Claude's memory across sessions**

_Claude fills this at the end of every session. The human saves it and uploads it at the start of every future session. This document is the only continuity between chats._

Keep the last 5 session entries. Move critical decisions to the Permanent Decisions section at the bottom. Delete older entries but never delete the Permanent Decisions.

---

## Project snapshot

| Field               | Value                                                                                                                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Project name        | My LLM from scratch                                                                                                                                                                                                      |
| Current phase       | Phase 4 — Transformer architecture (in progress)                                                                                                                                                                         |
| Overall progress    | 75% — Full Transformer block understood: self-attention, multi-head attention, feed-forward layer, layer norm, residual connections all covered. Decoder-only architecture (causal masking) remains before BUILD begins. |
| Last session date   | 9 April 2026                                                                                                                                                                                                             |
| Last session type   | LEARNING                                                                                                                                                                                                                 |
| Next session type   | BUILD                                                                                                                                                                                                                    |
| Next immediate task | Write `model/transformer.py` — the full Transformer block in code. Token embedding + positional encoding + N × (attention + FFN + layer norm + residual).                                                                |

---

## Phase and concept progress

| Phase                              | Status      | Notes                                                                                                                                                                                                                                                                          |
| ---------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Phase 1 — Python + math            | COMPLETE    | All core concepts understood. NumPy practised in code. Chain rule, functions, loops, classes done.                                                                                                                                                                             |
| Phase 2 — PyTorch fundamentals     | COMPLETE    | Tensors, autograd, forward pass, MSE loss, backward pass, training loop all covered and practised in code. Matplotlib plots added. Optimizer (SGD/Adam) not formally covered — introduced naturally in Phase 3.                                                                |
| Phase 3 — First neural network     | COMPLETE    | Neuron, nn.Module, nn.Linear, Adam, two-layer network, multiple input features, hidden layer representations, convergence, Softmax, cross-entropy loss, classification all understood. Two permanent decisions made: no Matplotlib from Session 8, GPU enabled from Session 8. |
| Phase 4 — Transformer architecture | IN PROGRESS | Tokenisation, embeddings, positional encoding, self-attention, multi-head attention, feed-forward layer, layer norm, and residual connections all understood. Full Transformer block concept complete. BUILD next.                                                             |
| Phase 5 — Dataset + pipeline       | NOT STARTED |                                                                                                                                                                                                                                                                                |
| Phase 6 — Train the model          | NOT STARTED |                                                                                                                                                                                                                                                                                |
| Phase 7 — Serve it locally         | NOT STARTED |                                                                                                                                                                                                                                                                                |

---

## Session entries

Most recent at the top. Keep last 5 entries only.

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
- Pre-LN ordering — UNDERSTOOD
- Full Transformer block — CAN EXPLAIN

**What is unfinished or unclear**

- Scaled dot-product (√d_k scaling) introduced but not yet practised in code
- Causal masking (decoder-only) not yet introduced — will cover at start of BUILD session before writing code
- All Phase 4 concepts now understood — ready for BUILD

**Exact next task**

BUILD session: Write `model/transformer.py` — the full Transformer block in code. Brief intro to causal masking at session start (5 minutes), then straight into code. Files to create: `model/transformer.py`.

---

### Session 9 — Embeddings and positional encoding

| Field              | Value        |
| ------------------ | ------------ |
| Date               | 8 April 2026 |
| Session type       | LEARNING     |
| Mode stayed clean? | Yes          |

**What was accomplished**

- Introduced tokenisation — text → integer IDs. Explained why raw integers cannot be fed to the network: they carry accidental numeric relationships (locker number analogy clicked immediately).
- Introduced the embedding table — `vocab_size × embedding_dim` matrix. Lookup by row index. Output shape: `(sequence_length, embedding_dim)`.
- Warm-up 1: human correctly derived output shape `(3, 6)` for input "ace" with vocab=10, dim=6. Explained the table structure without prompting.
- Introduced positional encoding — second learned embedding table for position indices. Added element-wise to token embeddings, not appended.
- Explained learned vs sinusoidal positional encoding. Decision: learned embeddings for this model (GPT-style). Sinusoidal not covered — not needed.
- Warm-up 2: human correctly derived full PE table shape `(128, 64)` and sliced input shape `(10, 64)`. Unprompted: identified that only the first 10 rows are used for a 10-token input. Correctly explained why element-wise addition requires identical shapes — reasoned from shape compatibility, not memory.
- Explained why the Transformer has no built-in sense of order — attention sees all positions simultaneously. Human said it felt like "nonsense that just somehow makes sense" — accepted as normal and resolved with the index card analogy.
- Full input pipeline now understood: token embeddings + positional encoding → `(seq_len, d_model)` matrix → enters Transformer.

**Concepts covered**

- Tokenisation / token IDs — UNDERSTOOD
- Embedding table — UNDERSTOOD
- Embedding shape — CAN EXPLAIN
- Learned embeddings trained by backprop — UNDERSTOOD
- Positional encoding — UNDERSTOOD
- Sequence length × embedding dim — CAN EXPLAIN
- Why Transformer has no built-in order — UNDERSTOOD

**What is unfinished or unclear**

- Sinusoidal positional encoding not covered — not needed for this model.
- Self-attention not yet introduced — covered Session 10.

**Exact next task**

LEARNING session: Self-attention. How each token decides how much to attend to every other token. Query, Key, Value introduced. Core mechanism of the Transformer.

---

### Session 8 — Softmax, cross-entropy loss, classification network

| Field              | Value        |
| ------------------ | ------------ |
| Date               | 8 April 2026 |
| Session type       | LEARNING     |
| Mode stayed clean? | Yes          |

**What was accomplished**

- Introduced Softmax — turns raw logits into probabilities. All outputs positive, all sum to 1.0, ranking preserved.
- Warm-up 1: human correctly predicted equal inputs `[3.0, 3.0, 3.0]` → equal outputs of 33.3% before any calculation. Correctly noted ranking is preserved.
- Introduced cross-entropy loss — `-log(probability of correct class)`. Punishes confident wrong answers hardest.
- Warm-up 2: given logits `[1.5, 0.2, 0.8]` with correct class B (index 1) — human correctly identified class A as most confident and that loss would be large.
- Explained `nn.CrossEntropyLoss` combines Softmax + log + negation in one step — never call Softmax manually during training.
- Built and ran `network3.py` — 2 inputs, 8 hidden neurons, 3-class output. Loss fell from 1.09 to 0.0003. Achieved 9/9 = 100% accuracy.
- Two permanent decisions made by human: no Matplotlib from this session onward, GPU enabled from this session onward.
- Explained standard GPU pattern: `device = torch.device(...)`, then `.to(device)` on inputs, labels, and model. Data and model must always be on the same device.
- Phase 3 fully closed.

**Concepts covered**

- Softmax — UNDERSTOOD
- Logits — UNDERSTOOD
- Cross-entropy loss — UNDERSTOOD
- `nn.CrossEntropyLoss` — UNDERSTOOD
- `torch.argmax` — UNDERSTOOD
- Accuracy — UNDERSTOOD
- Classification vs regression — UNDERSTOOD

**What is unfinished or unclear**

- Nothing. Phase 3 complete.

**Exact next task**

LEARNING session: Embeddings. How a token becomes a vector. First concept of Phase 4 — the Transformer architecture. GPU now active in all code from here.

---

### Session 7 — Multiple input features, hidden layer representations, convergence

| Field              | Value         |
| ------------------ | ------------- |
| Date               | 29 March 2026 |
| Session type       | LEARNING      |
| Mode stayed clean? | Yes           |

**What was accomplished**

- Introduced weighted sum (dot product) — each input gets its own weight, all multiplied and summed
- Warm-up: z = (2×3) + (−1×4) + 0.5 = 2.5, ReLU → 2.5 — solved correctly without help
- Built and ran `network2.py` — two-input network, 33 parameters, 1000 epochs, task: discover y = 2x₁ + 3x₂
- Predicted 5.0039 for test input x₁=1.0, x₂=1.0 (true: 5.0) — success
- Key misconception resolved through careful questioning: human confused inputs (x₁, x₂) with weights. Clarified: inputs are given to us and change every example; weights are what training finds and stay fixed after training
- Human independently ran 20000 epochs and observed weights barely changed — correctly reasoned the network had already converged
- Explained output layer weights as volume knobs — near-zero means the neuron is ignored
- Explained distributed representations — no single neuron holds the full answer; the 2:3 ratio is encoded across all neurons collectively
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

### Session 6 — Phase 3 start: neuron, nn.Module, first real network

| Field              | Value         |
| ------------------ | ------------- |
| Date               | 29 March 2026 |
| Session type       | LEARNING      |
| Mode stayed clean? | Yes           |

**What was accomplished**

- Introduced the full neuron: z = (weight × input) + bias, then output = ReLU(z)
- Explained bias as a learnable offset independent of input magnitude — temperature analogy used
- Explained why activation functions are necessary — without them, stacking layers collapses to a single linear operation regardless of depth
- Warm-up question: human computed both inputs correctly without help — z = −3 → ReLU → 0, and z = 1 → ReLU → 1
- Introduced `nn.Module` — PyTorch's base class that handles all parameter bookkeeping automatically
- Introduced `nn.Linear`, `nn.ReLU`, `nn.MSELoss`, `torch.optim.Adam`
- Wrote and ran `network.py` — two-layer network, 25 parameters, 500 epochs, task: learn y = 3x
- Loss fell from 12.4 to 0.004 — network learned the rule with small residual error
- Human correctly read tensor shape `[8, 1]` as "8 neurons, 1 input each" and distinguished it from `[1, 8]` unprompted — rows = neurons, columns = inputs

**Concepts covered**

- Bias — UNDERSTOOD
- Activation function / ReLU — UNDERSTOOD
- Why non-linearity matters — UNDERSTOOD
- `nn.Module`, `nn.Linear`, `nn.MSELoss` — UNDERSTOOD
- `torch.optim.Adam` — INTRODUCED
- Loss convergence behaviour (fast drop then slow tail) — UNDERSTOOD

**What is unfinished or unclear**

- Adam internals not covered — not needed yet
- Network only handles one input feature — scaled to multiple inputs Session 7
- GELU activation not yet introduced — covered Session 10

**Exact next task**

LEARNING session: multiple input features — what changes when a neuron receives many numbers instead of one.

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

---

_End of Document 3 — Session Log — v11_
