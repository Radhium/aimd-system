# DOC 3 — Session Log

**v14 · Claude's memory across sessions**

_Claude fills this at the end of every session. The human saves it and uploads it at the start of every future session. This document is the only continuity between chats._

Keep the last 5 session entries. Move critical decisions to the Permanent Decisions section at the bottom. Delete older entries but never delete the Permanent Decisions.

---

## Project snapshot

| Field               | Value                                                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Project name        | My LLM from scratch                                                                                                                |
| Current phase       | Phase 6 COMPLETE — model trains and generates text                                                                                 |
| Overall progress    | Full pipeline working: dataset → train → generate. Two training runs done. Output is recognisable Shakespearean dialogue.          |
| Last session date   | 10 April 2026                                                                                                                      |
| Last session type   | DEBUG + BUILD + EXPERIMENT                                                                                                         |
| Next session type   | REVIEW                                                                                                                             |
| Next immediate task | REVIEW session: assess output quality honestly, understand the train/val gap, decide what lever to pull next to improve the model. |

---

## Phase and concept progress

| Phase                              | Status      | Notes                                                                                                                                                                                                                                                                          |
| ---------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Phase 1 — Python + math            | COMPLETE    | All core concepts understood. NumPy practised in code. Chain rule, functions, loops, classes done.                                                                                                                                                                             |
| Phase 2 — PyTorch fundamentals     | COMPLETE    | Tensors, autograd, forward pass, MSE loss, backward pass, training loop all covered and practised in code. Matplotlib plots added. Optimizer (SGD/Adam) not formally covered — introduced naturally in Phase 3.                                                                |
| Phase 3 — First neural network     | COMPLETE    | Neuron, nn.Module, nn.Linear, Adam, two-layer network, multiple input features, hidden layer representations, convergence, Softmax, cross-entropy loss, classification all understood. Two permanent decisions made: no Matplotlib from Session 8, GPU enabled from Session 8. |
| Phase 4 — Transformer architecture | COMPLETE    | All concepts understood and all code written. transformer.py verified — 816,512 parameters, forward pass confirmed on GPU. Causal masking covered as final Learning concept. Phase 4 fully complete.                                                                           |
| Phase 5 — Dataset + pipeline       | COMPLETE    | dataset.py written and verified. train.py written and running. Full pipeline confirmed: data loads, model trains, checkpoints save correctly.                                                                                                                                  |
| Phase 6 — Train the model          | COMPLETE    | Two runs completed. 5,000 steps: val loss 1.96. 50,000 steps: val loss 1.49. generate.py written and working. Output is recognisable Shakespearean dialogue with correct structure and real character names.                                                                   |
| Phase 7 — Improve the model        | NOT STARTED | Next phase. Options: scale up model, add learning rate schedule, train longer. Decide in REVIEW session.                                                                                                                                                                       |

---

## Session entries

Most recent at the top. Keep last 5 entries only.

---

### Session 13 — DEBUG (train.py path errors) + BUILD (generate.py) + EXPERIMENT (50,000 steps)

| Field              | Value                      |
| ------------------ | -------------------------- |
| Date               | 10 April 2026              |
| Session type       | DEBUG → BUILD → EXPERIMENT |
| Mode stayed clean? | Yes                        |

**What was accomplished**

**DEBUG — train.py import errors:**

- Error 1: `ModuleNotFoundError: No module named 'data'` — ROOT pointed to `model/` instead of project root. Fix: wrapped `os.path.dirname(...)` one extra level.
- Error 2: `ModuleNotFoundError: No module named 'data.dataset'` — `dataset.py` was saved at `model/dataset.py` not `data/dataset.py`. Also `data/` had no `__init__.py`.
- Fix: `mv model/dataset.py data/dataset.py` + `touch data/__init__.py`. Both commands run in Git Bash.
- train.py ran successfully on next attempt.

**EXPERIMENT — First training run (5,000 steps):**

- Step 0 loss: ~82 (artefact — loss estimate before any training steps, normalises immediately)
- Final train loss: 1.8452, val loss: 1.9597, gap: 0.11
- Checkpoint saved to `runs/best_model.pt`
- Output at 5,000 steps: garbled words but correct play structure (character names, colons, line breaks)

**BUILD — generate.py:**

- Written to `model/generate.py`
- Loads checkpoint, encodes seed string, generates autoregressively with temperature sampling
- Temperature explained: < 1.0 = more focused, > 1.0 = more random. Default set to 0.8.
- Autoregressive generation: model's own output becomes its next input, trimmed to last SEQ_LEN tokens if context grows too long
- FutureWarning from torch.load noted — fix: add `weights_only=True`. Not urgent.
- Output at 5,000 steps confirmed working — structure correct, words garbled

**EXPERIMENT — Second training run (50,000 steps):**

- Final train loss: 1.2387, val loss: 1.4921 (best checkpoint), gap: ~0.25
- Train/val gap widened from 0.11 → 0.25 — early signs of overfitting
- Generated output: real Shakespeare character names (ROMEO, ESCALUS, PRINCE, First Servingman), correct dialogue formatting, real English words, Shakespearean rhythm — occasional invented words remain
- Simply training longer will eventually stop helping

**What is unfinished or unclear**

- `weights_only=True` fix not yet applied to generate.py (low priority — just a warning)
- Train/val gap widening — understood but not yet acted on

**Exact next task**

REVIEW session: assess what the model learned, understand the train/val gap and what it means for next steps, decide which lever to pull to improve output quality (scale up model / learning rate schedule / other).

---

### Session 12 — LEARNING (causal masking + Phase 5 concepts) + BUILD (dataset.py + train.py)

| Field              | Value                                                              |
| ------------------ | ------------------------------------------------------------------ |
| Date               | 10 April 2026                                                      |
| Session type       | LEARNING → BUILD (session ran long, two types covered)             |
| Mode stayed clean? | Mostly — session ran past token limit before train.py could be run |

**What was accomplished**

**LEARNING — Causal masking (Phase 4 final concept):**

- Explained the cheating problem: without masking, the model reads future tokens during training and learns nothing useful
- Showed the 4×4 attention score matrix before and after masking — upper triangle → -inf → 0 after Softmax
- Human correctly answered check question: e^(-inf) = 0, not just small — complete elimination, not deprioritisation
- Connected to existing code: make_causal_mask (torch.tril) and masked_fill already in transformer.py

**LEARNING — Phase 5 dataset pipeline concepts:**

- All four pipeline steps covered: get text → tokenize → split → serve batches
- Dataset decision: Tiny Shakespeare (human chose unprompted)
- Tokenizer mechanics: char_to_id and id_to_char, vocab_size derived from data
- x/y offset: human correctly answered — offset facilitates next-token prediction

**BUILD — dataset.py:**

- Written to `data/dataset.py` (originally at `dataset/dataset.py`, corrected in Session 13)
- Run successfully: vocab_size=65 ✓, shapes (4,128) ✓, x/y offset confirmed ✓

**BUILD — train.py:**

- Written to `model/train.py`
- Full training loop: hyperparameters → dataset → model → optimizer → train/eval loop → checkpoint saving
- model.train() / model.eval() pattern explained
- Session ended before first run

**Exact next task**

~~Run train.py. Report step 0 loss. Fill Experiment #1.~~ (Completed in Session 13)

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
- Total parameters: 816,512

**Exact next task**

~~LEARNING session: causal masking.~~ (Completed in Session 12)

---

### Session 10 — Self-attention, multi-head attention, feed-forward layer, layer norm, residual connections

| Field              | Value        |
| ------------------ | ------------ |
| Date               | 9 April 2026 |
| Session type       | LEARNING     |
| Mode stayed clean? | Yes          |

**What was accomplished**

- Self-attention, Q/K/V (library analogy), scaled dot-product attention, multi-head attention, feed-forward layer, GELU, layer normalisation, residual connections, Pre-LN ordering all covered
- Human completed all three warm-up derivations correctly
- Full Transformer block structure understood: `x = x + Attention(LayerNorm(x))` then `x = x + FFN(LayerNorm(x))`

**Exact next task**

~~BUILD: transformer.py.~~ (Completed in Session 11)

---

### Session 9 — Tokenisation, embeddings, positional encoding

| Field              | Value        |
| ------------------ | ------------ |
| Date               | 9 April 2026 |
| Session type       | LEARNING     |
| Mode stayed clean? | Yes          |

**What was accomplished**

- Tokenisation, token IDs ("locker numbers, not measurements"), embedding tables, positional encoding (learned, GPT-style) all covered
- Human correctly reasoned that position 0 always returns the same positional vector regardless of token identity

**Exact next task**

~~BUILD: transformer.py~~ → ~~LEARNING: self-attention first.~~ (Completed in Sessions 10–11)

---

## Permanent decisions archive

| Decision                                       | Made in session   | Reason — never change because...                                                                                                                                                                                                                                                       |
| ---------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Build on local PC only — no cloud              | Session 0 (setup) | The goal is to understand every part of the system. Cloud abstracts too much.                                                                                                                                                                                                          |
| Language: Python + PyTorch                     | Session 0 (setup) | Industry standard. Best learning resources. GPU support built in.                                                                                                                                                                                                                      |
| Use Python 3.11 venv, not system Python 3.14   | Session 1 (setup) | PyTorch has no CUDA build for Python 3.14. Always activate venv before any work. Command: C:\projects\myLLM\venv\Scripts\activate                                                                                                                                                      |
| Documents stored as .md files, not .docx       | Session 1 (setup) | Markdown is plain text — easier to read, edit, version, and paste into Claude.                                                                                                                                                                                                         |
| Character-level tokenizer                      | Session 1         | Simplest to implement, easiest to understand. Will upgrade to BPE in Phase 5 if needed.                                                                                                                                                                                                |
| Use VS Code play button to run files           | Session 3         | Ensures the correct venv Python is used. Running python directly in terminal can pick up wrong interpreter.                                                                                                                                                                            |
| Human copies code, does not type it            | Session 5         | Comments are half the learning. Human confirmed they would skip comments if writing themselves. Copying is faster and preserves the full annotated explanation.                                                                                                                        |
| No Matplotlib from Session 8 onward            | Session 8         | Human prefers print-only output — faster to run, less distraction. All loss tracking via print statements only.                                                                                                                                                                        |
| GPU enabled from Session 8 onward              | Session 8         | Phase 3 complete — all learning concepts done. Standard pattern every file must use: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, then `.to(device)` on inputs, labels, and model. Data and model must always be on the same device or PyTorch will error. |
| Learned positional embeddings (not sinusoidal) | Session 9         | GPT-style architecture. Same `nn.Embedding` as token embeddings but indexed by position. Simpler to implement and understand. Matches the reference architecture. Sinusoidal encoding is not needed for this project.                                                                  |
| GELU activation function                       | Session 10        | Standard for GPT-style Transformers. Smoother than ReLU — better gradient flow. Used in the feed-forward layer of every Transformer block.                                                                                                                                             |
| Pre-LN (layer norm before sub-blocks)          | Session 10        | Modern standard. LayerNorm applied before attention and before FFN. More stable during training than Post-LN.                                                                                                                                                                          |
| Residual connections throughout                | Session 10        | `x = x + block(x)` after every attention and FFN sub-block. Required for training stability. Non-negotiable in any Transformer.                                                                                                                                                        |
| Weight tying                                   | Session 11        | Output head shares weights with token embedding table. Standard GPT technique — reduces parameter count and is semantically consistent. Non-negotiable for this architecture.                                                                                                          |
| Dataset: Tiny Shakespeare                      | Session 12        | Standard reference for this architecture size. vocab_size=65 matches the placeholder already in transformer.py — no code changes needed.                                                                                                                                               |

---

_End of Document 3 — Session Log — v14_
