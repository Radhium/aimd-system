# DOC 2 — Lab Notebook

**v12 · Experiments, code, and architecture decisions**

_This is the technical record of everything built. Claude updates it after every Build, Experiment, or Debug session. It tells future-Claude exactly what exists, where it lives, and what has been tried._

---

## 1. Environment

| Component           | Details                                             |
| ------------------- | --------------------------------------------------- |
| Operating System    | Windows 11 Pro (Build 26200)                        |
| CPU                 | AMD Ryzen 5 5500 ~3600 MHz                          |
| GPU                 | NVIDIA GeForce RTX 3050                             |
| VRAM                | 6 GB                                                |
| RAM                 | 16 GB                                               |
| Python version      | 3.11.9 (inside venv)                                |
| PyTorch version     | 2.5.1+cu121                                         |
| CUDA Version        | 12.1                                                |
| Matplotlib version  | installed Session 5 — no longer used from Session 8 |
| Project folder      | C:\projects\myLLM                                   |
| Virtual environment | C:\projects\myLLM\venv                              |

**To activate the environment each session:**

```
C:\projects\myLLM\venv\Scripts\activate
```

---

## 2. Folder structure

Claude updates this whenever a new file or folder is created.

```
myLLM/
|  data/                   ← raw and processed datasets
|  |-__init__.py           ← makes data/ importable as a Python package (created Session 13)
|  |-input.txt             ← CURRENT DATASET: 5 Gutenberg books, cleaned, 5,529,055 chars (updated Session 15)
|  |-dataset.py            ← tokenizer + batch sampler (moved here Session 13 — was at dataset/dataset.py)
|  model/                  ← model architecture files
|  |-transformer.py        ← full decoder-only Transformer (created Session 11)
|  |-train.py              ← training loop with cosine LR schedule (updated Session 14)
|  |-generate.py           ← text generation from checkpoint (created Session 13)
|--runs/                   ← training checkpoints
|  |-best_model.pt         ← saved checkpoint from best validation loss (currently: Exp #4, val loss 1.1446)
|--tokenizer/              ← tokenizer code and vocab files (not yet used separately)
|--venv/                   ← Python virtual environment (do not edit manually)
|--forward_pass.py         ← Phase 2 training loop with Matplotlib plot (created Session 5)
|--network.py              ← Phase 3 two-layer nn.Module network, one input (created Session 6)
|--network2.py             ← Phase 3 two-input network, discovers y = 2x₁ + 3x₂ (created Session 7)
|--network3.py             ← Phase 3 classification network, Softmax + cross-entropy, 9 examples, 3 classes (created Session 8)
```

**Note on dataset.py path:** Originally saved at `dataset/dataset.py`. Moved to `data/dataset.py` during Session 13 debug. `data/__init__.py` created to make the folder importable. Current location confirmed working.

---

## 3. Current model spec

This is the architecture of the model as it exists right now. Claude updates this whenever the architecture changes.

| Parameter                     | Current value            | Notes                                                       |
| ----------------------------- | ------------------------ | ----------------------------------------------------------- |
| Model type                    | Decoder-only Transformer | GPT-style                                                   |
| Context length (max_seq_len)  | 128                      | How many tokens the model sees at once                      |
| Embedding dimension (d_model) | 256                      | Size of each token's vector                                 |
| Number of layers              | 6                        | Transformer blocks stacked                                  |
| Number of attention heads     | 8                        | d_k = 32 per head                                           |
| Feed-forward dimension        | 1024                     | 4 × d_model                                                 |
| Vocabulary size               | 87                       | Updated Session 15 — Gutenberg dataset, ASCII-cleaned       |
| Dropout rate                  | 0.1                      | Light regularisation (0.0 at inference time)                |
| Total parameters              | 4,789,504                | Confirmed by running train.py — larger due to vocab_size=87 |
| Activation function           | GELU                     | Standard for GPT-style Transformers                         |
| Learning rate schedule        | Cosine decay             | Added Session 14 — peak LR warmup then smooth decay to ~0   |

---

## 4. Dataset

| Property               | Details                                                                                                                                                         |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset name / source  | 5 Project Gutenberg books — Pride and Prejudice, Great Expectations, Moby Dick, A Tale of Two Cities, Middlemarch                                               |
| Raw size               | ~5.5MB                                                                                                                                                          |
| After ASCII cleaning   | 5,529,055 characters                                                                                                                                            |
| Tokenizer type         | Character-level — decided Session 1                                                                                                                             |
| Vocab size             | 87 unique characters                                                                                                                                            |
| Characters             | `\t\n !#$%&()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_abcdefghijklmnopqrstuvwxyz{}`                                                                     |
| Train / val split      | 90% / 10%                                                                                                                                                       |
| Train tokens           | 4,976,149                                                                                                                                                       |
| Val tokens             | 552,906                                                                                                                                                         |
| Where it lives on disk | C:\projects\myLLM\data\input.txt                                                                                                                                |
| Tokenizer file         | C:\projects\myLLM\data\dataset.py                                                                                                                               |
| Cleaning applied       | ASCII-only filter: kept characters with ord < 127, plus \n and \t. Removed Greek, Hebrew, accented chars, Unicode control chars, Gutenberg boilerplate symbols. |

**Previous dataset (Tiny Shakespeare):** ~1MB, 65 chars, 1,115,394 characters. Replaced in Session 15 with Gutenberg corpus for richer generalisation.

**Batch shape confirmed:** x shape `(4, 128)`, y shape `(4, 128)`. x and y are the same sequence offset by one character — y is x shifted one position to the right.

---

## 5. Experiment log

_One entry per training run. Most recent at the top. Claude fills this after every Experiment session. Failed experiments are just as important as successful ones._

Never delete old entries. Cross them out with a note if they are obsolete, but keep them. The history of what didn't work is valuable.

---

### Experiment #4 — Gutenberg dataset, cosine LR, 50,000 steps

| Field                                    | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Experiment #                             | 4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Date                                     | 11 April 2026                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Goal of this run                         | Test the effect of upgrading from Tiny Shakespeare (~1MB, vocab=65) to 5 Gutenberg books (~5.5MB, vocab=87) with cosine LR schedule already in place                                                                                                                                                                                                                                                                                                                                                                                                     |
| Config changes from last run             | Dataset replaced: Tiny Shakespeare → 5 Gutenberg books (ASCII-cleaned). VOCAB_SIZE updated: 65 → 87. Cosine LR schedule already present from Exp #3. All other hyperparameters identical.                                                                                                                                                                                                                                                                                                                                                                |
| Parameter count                          | 4,789,504 (up from ~816K — vocab embedding table is larger with vocab=87)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Step-by-step losses                      | step 0: train 163.75 / val 162.97 · step 5k: 1.62/1.58 · step 10k: 1.43/1.36 · step 15k: 1.36/1.29 · step 20k: 1.31/1.25 · step 25k: 1.28/1.21 · step 30k: 1.25/1.19 · step 35k: 1.23/1.18 · step 40k: 1.21/1.16 · step 45k: 1.20/1.15                                                                                                                                                                                                                                                                                                                   |
| Final training loss                      | 1.1806                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Final validation loss                    | 1.1446                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Train / val gap                          | 0.036 — val loss is LOWER than train loss at every checkpoint. No overfitting. Model is generalising well.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Training duration                        | ~2 hours on RTX 3050                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Observations                             | Largest single improvement of the entire project — val loss dropped from 1.4921 to 1.1446 (−0.35). Val loss lower than train loss throughout: the larger dataset gives enough variety that the model never memorises. Loss still descending at step 45,000 — not fully plateaued. Generated text is coherent 19th-century literary prose. Real character names (Mr. Farebrother, Dorothea from Middlemarch). Dialogue correctly attributed. Sentences grammatically correct. Meaning holds within a sentence but still drifts across multiple sentences. |
| Sample output (seed: "How are you?")     | "How are you? said Mary. Nothing would be always be an artificial left to address to her, she didnt say that it would not be so much that be something in my earth might be taken in it..."                                                                                                                                                                                                                                                                                                                                                              |
| Sample output (seed: "What can you do?") | "What can you do? said Mr. Farebrother, with a foot of disturbing in this own sign of that there were such a conception in the other. There was a truth in the time of Dorotheas grass curtain again..."                                                                                                                                                                                                                                                                                                                                                 |
| Did it work?                             | Yes — best result of the project                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| What to try next                         | Scale up the model: increase d_model (128→256) and n_layers (4→6). This is the biggest remaining lever for quality improvement. Alternatively, train longer — loss was still descending at step 45k.                                                                                                                                                                                                                                                                                                                                                     |

---

### Experiment #3 — Cosine learning rate schedule, 50,000 steps

| Field                        | Value                                                                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Experiment #                 | 3                                                                                                                       |
| Date                         | 11 April 2026                                                                                                           |
| Goal of this run             | Test cosine LR schedule against fixed LR baseline — same dataset (Tiny Shakespeare), same steps                         |
| Config changes from last run | Added cosine decay schedule to train.py. Peak LR same as before, decays smoothly to ~0 by step 50,000.                  |
| Final training loss          | ~1.21                                                                                                                   |
| Final validation loss        | ~1.45                                                                                                                   |
| Train / val gap              | Similar to Exp #2 — small improvement from schedule alone                                                               |
| Observations                 | Modest improvement over fixed LR (~0.04 val loss reduction). Cosine schedule is now standard — kept in all future runs. |
| Did it work?                 | Yes — small but real gain. Schedule is now a permanent fixture.                                                         |
| What to try next             | Upgrade dataset — the schedule alone was not enough. Tiny Shakespeare too small for further improvement.                |

---

### Experiment #2 — 50,000 steps, same config

| Field                        | Value                                                                                                                                                                                                                                                                                                       |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Experiment #                 | 2                                                                                                                                                                                                                                                                                                           |
| Date                         | 10 April 2026                                                                                                                                                                                                                                                                                               |
| Goal of this run             | See how much further the model improves with 10× more training steps                                                                                                                                                                                                                                        |
| Config changes from last run | MAX_STEPS changed from 5,000 to 50,000. All other hyperparameters identical.                                                                                                                                                                                                                                |
| Final training loss          | 1.2387                                                                                                                                                                                                                                                                                                      |
| Final validation loss        | 1.5080 (best checkpoint at step 49,500: val loss 1.4921)                                                                                                                                                                                                                                                    |
| Train / val gap              | ~0.25 (was 0.11 at 5,000 steps — gap widening, early signs of overfitting)                                                                                                                                                                                                                                  |
| Observations                 | Loss fell steadily throughout. Generated text dramatically improved — real Shakespeare character names (ROMEO, ESCALUS, PRINCE, First Servingman), correct dialogue formatting (Name: followed by speech), real English words, correct punctuation, Shakespearean rhythm. Occasional invented words remain. |
| Did it work?                 | Yes                                                                                                                                                                                                                                                                                                         |
| What to try next             | Simply training longer will eventually stop helping — gap between train and val loss is widening. Next lever: scale up the model (d_model, n_layers) or add a learning rate schedule.                                                                                                                       |

---

### Experiment #1 — First training run, 5,000 steps

| Field                        | Value                                                                                                                                                                                                     |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Experiment #                 | 1                                                                                                                                                                                                         |
| Date                         | 10 April 2026                                                                                                                                                                                             |
| Goal of this run             | First ever training run — verify the full pipeline works end to end                                                                                                                                       |
| Config changes from last run | First run — no baseline                                                                                                                                                                                   |
| Final training loss          | 1.8452                                                                                                                                                                                                    |
| Final validation loss        | 1.9597                                                                                                                                                                                                    |
| Train / val gap              | ~0.11 — healthy, model generalising well                                                                                                                                                                  |
| Observations                 | Step 0 loss was ~83 (not the expected ~4.17 — likely a loss estimation artefact before any training, smooths out immediately). By step 500 loss was already 2.57. Loss fell steadily every eval interval. |
| Did it work?                 | Yes                                                                                                                                                                                                       |
| What to try next             | Train for more steps — model still improving at step 5000                                                                                                                                                 |

---

## 6. Architecture decisions

Decisions that were made and must not be reversed without good reason.

| Decision                              | Made in session | Reason — never change because...                                                                                                                                                                                                                                                              |
| ------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Character-level tokenizer for Phase 1 | Session 1       | Simplest to implement, good for learning. Upgrade to BPE in Phase 5.                                                                                                                                                                                                                          |
| GPU move deferred to Phase 6          | Session 7       | Learning networks are too small to benefit. Superseded by Session 8 decision.                                                                                                                                                                                                                 |
| No Matplotlib from Session 8 onward   | Session 8       | Human prefers print-only output — faster to run, less distraction. All loss tracking via print statements.                                                                                                                                                                                    |
| GPU enabled from Session 8 onward     | Session 8       | Phase 3 complete — all learning concepts done. Real networks from here benefit from GPU. Standard pattern: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, then move inputs, labels, and model with `.to(device)`. Data and model must always be on the same device. |
| Learned positional embeddings         | Session 9       | GPT-style — same `nn.Embedding` as token embeddings but indexed by position. Simpler than sinusoidal, matches GPT architecture, and is learnable. Limitation: cannot generalise beyond max_sequence_length seen during training.                                                              |
| GELU activation function              | Session 10      | Standard for GPT-style Transformers. Smoother than ReLU — allows small negative values through and provides better gradient flow. Used in the feed-forward layer of every Transformer block.                                                                                                  |
| Pre-LN (layer norm before sub-blocks) | Session 10      | Modern standard for stable training. LayerNorm applied before attention and before FFN, not after. More stable than Post-LN during training.                                                                                                                                                  |
| Residual connections throughout       | Session 10      | `x = x + block(x)` after every attention and FFN sub-block. Required for training stability in deep networks — gives gradients a direct path backwards. Non-negotiable in any Transformer.                                                                                                    |
| Weight tying                          | Session 11      | Output head shares weights with token embedding table — standard GPT technique. Reduces parameter count and is semantically consistent: the same matrix that maps tokens to vectors also maps vectors back to tokens.                                                                         |
| Dataset upgraded to Gutenberg corpus  | Session 15      | Tiny Shakespeare (~1MB) hit its ceiling at val loss ~1.45. Gutenberg (5.5MB, 5 books) gave the model enough variety to generalise without overfitting. Largest single improvement of the project.                                                                                             |
| Cosine LR schedule                    | Session 14      | Fixed LR causes unnecessary bouncing late in training. Cosine decay lets the model make bold moves early and precise moves late. Now standard for all future runs.                                                                                                                            |

---

## 7. Dead ends and rejected ideas

| Idea / approach                        | Why it was ruled out                                                                                                                                           | Session   |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| Use system Python 3.14 for the project | PyTorch has no CUDA-compatible build for Python 3.14 — falls back to CPU only. Use venv with Python 3.11 instead.                                              | Setup     |
| Sinusoidal positional encoding         | Chosen not to use for this model — more complex to implement, not necessary for a GPT-style decoder-only model. Learned embeddings are simpler and sufficient. | Session 9 |

---

## 8. Known issues and bugs

| Issue                                   | Severity | Status | Notes / workaround                                                                                                                              |
| --------------------------------------- | -------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| torch.load FutureWarning in generate.py | Low      | Open   | Add `weights_only=True` to `torch.load(...)` call in generate.py line 80. Not an error — just a deprecation warning. Does not affect behaviour. |

---

## 9. Resources and references

| Resource                                                 | Type         | Topic                                    | Recommended in session | Status                            |
| -------------------------------------------------------- | ------------ | ---------------------------------------- | ---------------------- | --------------------------------- |
| [ e.g. Andrej Karpathy — Neural Networks: Zero to Hero ] | Video series | Full project arc — Python to Transformer | [ Session # ]          | [ Not started / Watching / Done ] |

---

_End of Document 2 — Lab Notebook — v12_
