# DOC 3 — Session Log

**v15 · Claude's memory across sessions**

_Claude fills this at the end of every session. The human saves it and uploads it at the start of every future session. This document is the only continuity between chats._

Keep the last 5 session entries. Move critical decisions to the Permanent Decisions section at the bottom. Delete older entries but never delete the Permanent Decisions.

---

## Project snapshot

| Field             | Value                                                                                                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Project name      | My LLM from scratch                                                                                                                                                                                          |
| Current phase     | Phase 7 — Improving the model (in progress)                                                                                                                                                                  |
| Overall progress  | 4 training experiments complete. Dataset upgraded to 5 Gutenberg books (5.5MB). Best val loss: 1.1446. Generated text is coherent 19th-century literary prose with correct grammar and real character names. |
| Last session date | 11 April 2026                                                                                                                                                                                                |
| Last session type | EXPERIMENT + REVIEW                                                                                                                                                                                          |

---

## Phase and concept progress

| Phase                              | Status   | Notes                                                                                                                                                                                                                                                                          |
| ---------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Phase 1 — Python + math            | COMPLETE | All core concepts understood. NumPy practised in code. Chain rule, functions, loops, classes done.                                                                                                                                                                             |
| Phase 2 — PyTorch fundamentals     | COMPLETE | Tensors, autograd, forward pass, MSE loss, backward pass, training loop all covered and practised in code. Matplotlib plots added. Optimizer (SGD/Adam) not formally covered — introduced naturally in Phase 3.                                                                |
| Phase 3 — First neural network     | COMPLETE | Neuron, nn.Module, nn.Linear, Adam, two-layer network, multiple input features, hidden layer representations, convergence, Softmax, cross-entropy loss, classification all understood. Two permanent decisions made: no Matplotlib from Session 8, GPU enabled from Session 8. |
| Phase 4 — Transformer architecture | COMPLETE | All concepts understood and all code written. transformer.py verified — 816,512 parameters, forward pass confirmed on GPU. Causal masking covered as final Learning concept. Phase 4 fully complete.                                                                           |
| Phase 5 — Dataset + pipeline       | COMPLETE | dataset.py written and verified. train.py written and running. Full pipeline confirmed: data loads, model trains, checkpoints save correctly.                                                                                                                                  |
| Phase 6 — Train the model          | COMPLETE | Two runs completed. 5,000 steps: val loss 1.96. 50,000 steps: val loss 1.49. generate.py written and working. Output is recognisable Shakespearean dialogue with correct structure and real character names.                                                                   |
| Phase 7 — Improve the model        | COMPLETE | Exp #3: cosine LR schedule (small gain). Exp #4: Gutenberg dataset upgrade — val loss 1.1446, biggest improvement of the project. Next: scale up model architecture.                                                                                                           |

---

## Session entries

Most recent at the top. Keep last 5 entries only.

---

### Session 15 — EXPERIMENT: Gutenberg dataset + REVIEW

| Field              | Value               |
| ------------------ | ------------------- |
| Date               | 11 April 2026       |
| Session type       | EXPERIMENT → REVIEW |
| Mode stayed clean? | Yes                 |

**What was accomplished**

**Dataset upgrade:**

- Replaced Tiny Shakespeare (~1MB, vocab=65) with 5 Project Gutenberg books: Pride and Prejudice, Great Expectations, Moby Dick, A Tale of Two Cities, Middlemarch
- Raw combined size: ~5.5MB. After ASCII cleaning (ord < 127 filter): 5,529,055 characters
- Vocab size: 87 (was 122 before cleaning — removed Greek, Hebrew, accented chars, Unicode control chars, Gutenberg boilerplate symbols)
- VOCAB_SIZE updated to 87 in train.py

**Experiment #4 — results:**

- 50,000 steps, cosine LR, vocab=87, same architecture
- Parameter count: 4,789,504 (larger embedding table due to vocab=87)
- Final train loss: 1.1806 / val loss: 1.1446
- Val loss was LOWER than train loss at every checkpoint — no overfitting. Larger dataset gave enough variety.
- Loss still descending at step 45,000 — not fully plateaued
- Training time: ~2 hours on RTX 3050

**REVIEW — quality assessment:**

- Generated text is coherent 19th-century literary prose — real character names (Mr. Farebrother, Dorothea from Middlemarch), correct dialogue attribution, grammatically correct sentences
- Meaning holds within a sentence but drifts across multiple sentences — expected at this scale
- This is a qualitatively different model from the Shakespeare one: has learned sentence skeleton, not just surface costume
- Val loss improvement: 1.4921 → 1.1446 (−0.35) — single biggest gain of the project

**What is unfinished or unclear**

- Loss was still descending at step 45k — training longer may yield more, but scaling is a bigger lever

---

### Session 14 — EXPERIMENT: Cosine LR schedule

| Field              | Value         |
| ------------------ | ------------- |
| Date               | 11 April 2026 |
| Session type       | EXPERIMENT    |
| Mode stayed clean? | Yes           |

**What was accomplished**

- Added cosine learning rate schedule to train.py
- Schedule: LR starts at peak (same as before), decays smoothly following cosine curve to near-zero by step 50,000
- Warmup period included: LR ramps up linearly for first N steps before cosine decay begins
- Re-ran 50,000 steps on Tiny Shakespeare — small but real improvement over fixed LR baseline
- Cosine schedule now permanent for all future runs

**What is unfinished or unclear**

- Small gain from schedule alone confirmed — dataset upgrade was the needed next step

**Exact next task**
~~EXPERIMENT: upgrade dataset to Gutenberg books.~~ (Completed in Session 15)

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

- Final train loss: 1.8452, val loss: 1.9597, gap: 0.11
- Checkpoint saved to `runs/best_model.pt`

**BUILD — generate.py:**

- Written to `model/generate.py`
- Loads checkpoint, encodes seed string, generates autoregressively with temperature sampling
- FutureWarning from torch.load noted — fix: add `weights_only=True`. Not urgent.

**EXPERIMENT — Second training run (50,000 steps):**

- Final train loss: 1.2387, val loss: 1.4921 (best checkpoint), gap: ~0.25
- Train/val gap widened — early signs of overfitting on small dataset

**Exact next task**
~~REVIEW session.~~ (Completed in Session 14/15)

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
| Dataset: Gutenberg corpus (5 books)            | Session 15        | Tiny Shakespeare (~1MB) hit ceiling at val loss ~1.45. Gutenberg (5.5MB) gave enough variety to generalise without overfitting. Largest single improvement of the project (−0.35 val loss). input.txt now contains 5 books, ASCII-cleaned, vocab=87.                                   |
| Cosine LR schedule                             | Session 14        | Fixed LR causes unnecessary bouncing late in training. Cosine decay: bold moves early, precise moves late. Now standard for all future runs.                                                                                                                                           |

---

_End of Document 3 — Session Log — v15_
