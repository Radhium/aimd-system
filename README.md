# aimd-system

**AI-Assisted Machine learning Documentation system** — a structured, session-based learning OS for building a Transformer language model from scratch.

---

## What this is

This repo is the complete archive of a solo project to build a GPT-style language model from the ground up — understanding every layer of it, not just running it.

It started with "what is a derivative" and ended with a trained Transformer with ~3.2M parameters, a cosine learning rate schedule, and scaling experiments across multiple runs.

**Duration:** 17 days · **Sessions:** 12 · **Commits:** 16

---

## The model

A decoder-only Transformer (GPT-style) implemented in pure PyTorch.

| Hyperparameter | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| d_model | 256 |
| Attention heads | 8 |
| Layers | 6 |
| FFN dim | 1024 |
| Max sequence length | 128 |
| Parameters | ~3.2M |

Key implementation details:
- Multi-head self-attention with causal masking
- Pre-LayerNorm (more stable than Post-LN)
- GELU activation in the feed-forward network
- Learned positional embeddings
- Weight tying between token embedding and output head
- Cosine learning rate decay

---

## Repo structure

```
aimd-system/
│
├── myLLM/                  # The final model codebase
│   ├── model/
│   │   ├── transformer.py  # Full model architecture
│   │   ├── train.py        # Training loop
│   │   └── generate.py     # Text generation
│   ├── data/
│   │   ├── dataset.py      # Data pipeline
│   │   └── input.txt       # Training corpus
│   └── runs/
│       └── best_model.pt   # Saved model checkpoint
│
├── continue/               # Full session archive
│   ├── s1/ … s12/          # One folder per session
│   │   ├── Chat.md         # Full conversation log
│   │   ├── DOC1_*.md       # Concept map at that point
│   │   ├── DOC2_*.md       # Lab notebook at that point
│   │   └── DOC3_*.md       # Session log at that point
│
├── DOC0_System_Rules_v2.md # The rules that governed every session
├── DOC1_Concept_Map_v2.md  # Initial concept map (see s12/ for latest)
├── DOC2_Lab_Notebook_v2.md # Initial lab notebook
└── DOC3_Session_Log_v2.md  # Initial session log
```

---

## The learning arc

Every commit message is a concept map of that session's content.

| Session | What was covered |
|---|---|
| S1 | NumPy, matrix multiplication, derivatives, gradients |
| S2 | Chain rule, Python functions/loops/classes |
| S3 | Tensors, autograd, computational graph |
| S4 | Forward pass, loss (MSE), backward pass, training loop |
| S5 | Bias, ReLU, non-linearity, nn.Module, Adam, MSELoss |
| S6 | Hidden layers, convergence, GPU vs CPU |
| S7 | Softmax, cross-entropy loss |
| S8 | Tokenisation, embeddings, positional encoding |
| S9 | Self-attention, Q/K/V, scaled dot-product, multi-head, GELU, LayerNorm, residual connections |
| S10 | Built transformer.py |
| S11 | Structured into myLLM/ project |
| S12 | Cosine LR schedule, scaling experiments (Expt-3 → Expt-5) |

---

## The system

This project ran on a structured session protocol called **AIMD** (AI-assisted Machine-learning Documentation). Each session had a declared type (LEARNING / BUILD / EXPERIMENT / DEBUG / REVIEW), a defined goal, and ended with updated documentation. Claude acted as tutor, architect, and debugger — with no memory between sessions. The four DOC files were the entire shared memory.

---

## Hardware

Trained locally on a single mid-range RTX GPU. No cloud, no paid APIs.
