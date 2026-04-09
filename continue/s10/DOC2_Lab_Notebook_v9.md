# DOC 2 — Lab Notebook

**v9 · Experiments, code, and architecture decisions**

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
  data/                 ← raw and processed datasets
  tokenizer/            ← tokenizer code and vocab files
  model/                ← model architecture files
    config.py           ← hyperparameters (not created yet)
    transformer.py      ← full decoder-only Transformer (created Session 11)
  runs/                 ← training logs and checkpoints
  venv/                 ← Python virtual environment (do not edit manually)
  forward_pass.py       ← Phase 2 training loop with Matplotlib plot (created Session 5)
  network.py            ← Phase 3 two-layer nn.Module network, one input (created Session 6)
  network2.py           ← Phase 3 two-input network, discovers y = 2x₁ + 3x₂ (created Session 7)
  network3.py           ← Phase 3 classification network, Softmax + cross-entropy, 9 examples, 3 classes (created Session 8)
  train.py              ← training loop (not created yet — forward_pass.py is the learning version)
  eval.py               ← evaluation / inference (not created yet)
```

---

## 3. Current model spec

This is the architecture of the model as it exists right now. Claude updates this whenever the architecture changes.

| Parameter                     | Current value                  | Notes                                                    |
| ----------------------------- | ------------------------------ | -------------------------------------------------------- |
| Model type                    | Decoder-only Transformer       | GPT-style                                                |
| Context length (max_seq_len)  | 128                            | How many tokens the model sees at once                   |
| Embedding dimension (d_model) | 128                            | Size of each token's vector                              |
| Number of layers              | 4                              | Transformer blocks stacked                               |
| Number of attention heads     | 4                              | d_k = 32 per head                                        |
| Feed-forward dimension        | 512                            | 4 × d_model                                              |
| Vocabulary size               | NOT SET — placeholder 65       | Real value set in Phase 5 when dataset is chosen         |
| Dropout rate                  | 0.1                            | Light regularisation                                     |
| Total parameters              | 816,512                        | Confirmed by running transformer.py on GPU               |
| Activation function           | GELU                           | Standard for GPT-style Transformers                      |

---

## 4. Dataset

| Property               | Details                                     |
| ---------------------- | ------------------------------------------- |
| Dataset name / source  | NOT SET YET                                 |
| Raw size               | NOT SET YET                                 |
| After cleaning         | NOT SET YET                                 |
| Tokenizer type         | Character-level — decided Session 1         |
| Vocab size             | NOT SET YET — depends on dataset characters |
| Train / val split      | NOT SET YET — e.g. 90% / 10%                |
| Where it lives on disk | NOT SET YET                                 |

---

## 5. Experiment log

_One entry per training run. Most recent at the top. Claude fills this after every Experiment session. Failed experiments are just as important as successful ones._

Never delete old entries. Cross them out with a note if they are obsolete, but keep them. The history of what didn't work is valuable.

### Experiment template — copy this for each run

| Field                        | Value              |
| ---------------------------- | ------------------ |
| Experiment #                 |                    |
| Date                         |                    |
| Goal of this run             |                    |
| Config changes from last run |                    |
| Final training loss          |                    |
| Final validation loss        |                    |
| Observations                 |                    |
| Did it work?                 | Yes / No / Partial |
| What to try next             |                    |

---

## 6. Architecture decisions

Decisions that were made and must not be reversed without good reason. Claude adds entries here when a significant architectural or technical choice is made.

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
| Weight tying                          | Session 11      | Output head shares weights with token embedding table — standard GPT technique. Reduces parameter count and is semantically consistent: the same matrix that maps tokens to vectors also maps vectors back to tokens.                                                                          |

---

## 7. Dead ends and rejected ideas

_Things that were tried or considered and ruled out. Claude fills this during Debug and Review sessions._

| Idea / approach                        | Why it was ruled out                                                                                                                                           | Session   |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| Use system Python 3.14 for the project | PyTorch has no CUDA-compatible build for Python 3.14 — falls back to CPU only. Use venv with Python 3.11 instead.                                              | Setup     |
| Sinusoidal positional encoding         | Chosen not to use for this model — more complex to implement, not necessary for a GPT-style decoder-only model. Learned embeddings are simpler and sufficient. | Session 9 |

---

## 8. Known issues and bugs

| Issue         | Severity | Status | Notes / workaround |
| ------------- | -------- | ------ | ------------------ |
| No issues yet | —        | —      | —                  |

---

## 9. Resources and references

| Resource                                                 | Type         | Topic                                    | Recommended in session | Status                            |
| -------------------------------------------------------- | ------------ | ---------------------------------------- | ---------------------- | --------------------------------- |
| [ e.g. Andrej Karpathy — Neural Networks: Zero to Hero ] | Video series | Full project arc — Python to Transformer | [ Session # ]          | [ Not started / Watching / Done ] |

---

_End of Document 2 — Lab Notebook — v9_
