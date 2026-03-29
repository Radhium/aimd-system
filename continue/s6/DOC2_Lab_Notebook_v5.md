# DOC 2 — Lab Notebook

**v5 · Experiments, code, and architecture decisions**

_This is the technical record of everything built. Claude updates it after every Build, Experiment, or Debug session. It tells future-Claude exactly what exists, where it lives, and what has been tried._

---

## 1. Environment

| Component           | Details                      |
| ------------------- | ---------------------------- |
| Operating System    | Windows 11 Pro (Build 26200) |
| CPU                 | AMD Ryzen 5 5500 ~3600 MHz   |
| GPU                 | NVIDIA GeForce RTX 3050      |
| VRAM                | 6 GB                         |
| RAM                 | 16 GB                        |
| Python version      | 3.11.9 (inside venv)         |
| PyTorch version     | 2.5.1+cu121                  |
| CUDA Version        | 12.1                         |
| Matplotlib version  | installed Session 5          |
| Project folder      | C:\projects\myLLM            |
| Virtual environment | C:\projects\myLLM\venv       |

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
    transformer.py      ← the model itself (not created yet)
  runs/                 ← training logs and checkpoints
  venv/                 ← Python virtual environment (do not edit manually)
  forward_pass.py       ← Phase 2 training loop with Matplotlib plot (created Session 5)
  network.py            ← Phase 3 two-layer nn.Module network, one input (created Session 6)
  network2.py           ← Phase 3 two-input network, discovers y = 2x₁ + 3x₂ (created Session 7)
  train.py              ← training loop (not created yet — forward_pass.py is the learning version)
  eval.py               ← evaluation / inference (not created yet)
```

---

## 3. Current model spec

This is the architecture of the model as it exists right now. Claude updates this whenever the architecture changes.

| Parameter                     | Current value | Notes                                      |
| ----------------------------- | ------------- | ------------------------------------------ |
| Model type                    | NOT SET YET   | Decoder-only Transformer (GPT style)       |
| Context length (block size)   | NOT SET YET   | How many tokens the model sees at once     |
| Embedding dimension (d_model) | NOT SET YET   | Size of each token's vector representation |
| Number of layers              | NOT SET YET   | How many Transformer blocks stacked        |
| Number of attention heads     | NOT SET YET   | Must divide evenly into d_model            |
| Feed-forward dimension        | NOT SET YET   | Usually 4 × d_model                        |
| Vocabulary size               | NOT SET YET   | Number of unique tokens                    |
| Dropout rate                  | NOT SET YET   | Regularization — usually 0.1–0.2           |
| Total parameters              | NOT SET YET   | Calculate after architecture is set        |
| Activation function           | NOT SET YET   | Usually GELU for transformers              |

---

## 4. Dataset

| Property               | Details                                             |
| ---------------------- | --------------------------------------------------- |
| Dataset name / source  | NOT SET YET                                         |
| Raw size               | NOT SET YET                                         |
| After cleaning         | NOT SET YET                                         |
| Tokenizer type         | NOT SET YET — e.g. character-level / BPE / tiktoken |
| Vocab size             | NOT SET YET                                         |
| Train / val split      | NOT SET YET — e.g. 90% / 10%                        |
| Where it lives on disk | NOT SET YET                                         |

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

| Decision                              | Made in session | Reason — never change because...                                                                  |
| ------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------- |
| Character-level tokenizer for Phase 1 | [ Session # ]   | Simplest to implement, good for learning. Upgrade to BPE in Phase 5.                              |
| GPU move deferred to Phase 6          | Session 7       | Learning networks are too small to benefit. Add `.to("cuda")` properly in the real training loop. |
| [ add more as project grows ]         |                 |                                                                                                   |

---

## 7. Dead ends and rejected ideas

_Things that were tried or considered and ruled out. Claude fills this during Debug and Review sessions._

| Idea / approach                        | Why it was ruled out                                                                                              | Session |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------- |
| Use system Python 3.14 for the project | PyTorch has no CUDA-compatible build for Python 3.14 — falls back to CPU only. Use venv with Python 3.11 instead. | Setup   |
| [ add more as project grows ]          |                                                                                                                   |         |

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
| [ add more as project grows ]                            |              |                                          |                        |                                   |

---

_End of Document 2 — Lab Notebook — v5_
