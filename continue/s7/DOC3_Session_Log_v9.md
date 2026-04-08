# DOC 3 — Session Log

**v9 · Claude's memory across sessions**

_Claude fills this at the end of every session. The human saves it and uploads it at the start of every future session. This document is the only continuity between chats._

Keep the last 5 session entries. Move critical decisions to the Permanent Decisions section at the bottom. Delete older entries but never delete the Permanent Decisions.

---

## Project snapshot

| Field               | Value                                                                                                                                        |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Project name        | My LLM from scratch                                                                                                                          |
| Current phase       | Phase 4 — Transformer architecture (not started)                                                                                             |
| Overall progress    | 40% — Phase 3 complete. All neural network basics understood including classification, Softmax, and cross-entropy loss.                      |
| Last session date   | 8 April 2026                                                                                                                                 |
| Last session type   | LEARNING                                                                                                                                     |
| Next session type   | LEARNING — Phase 4 start                                                                                                                     |
| Next immediate task | Embeddings — how a token (a character or word) becomes a vector the network can work with. First concept of the Transformer. GPU now active. |

---

## Phase and concept progress

| Phase                              | Status      | Notes                                                                                                                                                                                                                                                                          |
| ---------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Phase 1 — Python + math            | COMPLETE    | All core concepts understood. NumPy practised in code. Chain rule, functions, loops, classes done.                                                                                                                                                                             |
| Phase 2 — PyTorch fundamentals     | COMPLETE    | Tensors, autograd, forward pass, MSE loss, backward pass, training loop all covered and practised in code. Matplotlib plots added. Optimizer (SGD/Adam) not formally covered — introduced naturally in Phase 3.                                                                |
| Phase 3 — First neural network     | COMPLETE    | Neuron, nn.Module, nn.Linear, Adam, two-layer network, multiple input features, hidden layer representations, convergence, Softmax, cross-entropy loss, classification all understood. Two permanent decisions made: no Matplotlib from Session 8, GPU enabled from Session 8. |
| Phase 4 — Transformer architecture | NOT STARTED |                                                                                                                                                                                                                                                                                |
| Phase 5 — Dataset + pipeline       | NOT STARTED |                                                                                                                                                                                                                                                                                |
| Phase 6 — Train the model          | NOT STARTED |                                                                                                                                                                                                                                                                                |
| Phase 7 — Serve it locally         | NOT STARTED |                                                                                                                                                                                                                                                                                |

---

## Session entries

Most recent at the top. Keep last 5 entries only.

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
- What a hidden layer learns (distributed representations) — UNDERSTOOD
- Convergence (empirically discovered) — UNDERSTOOD
- Output layer as weighted combination of hidden signals — UNDERSTOOD
- GPU vs CPU default in PyTorch — UNDERSTOOD

**What is unfinished or unclear**

- Softmax and cross-entropy loss not yet covered — done in Session 8
- GPU integration deferred — now active from Session 8

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
- GELU activation not yet introduced — comes in Phase 4

**Exact next task**

LEARNING session: multiple input features — what changes when a neuron receives many numbers instead of one.

---

### Session 5 — Forward pass, loss, backward pass, training loop

| Field              | Value         |
| ------------------ | ------------- |
| Date               | 29 March 2026 |
| Session type       | LEARNING      |
| Mode stayed clean? | Yes           |

**What was accomplished**

- Explained forward pass, MSE loss, backward pass, full training loop
- Built `forward_pass.py` with Matplotlib plots
- Key misconception corrected: human thought weight 0.5 was the input data
- Human independently changed print interval, noticed loss showed 0.0000 while weight was 2.9999
- Two charts rendered correctly in VS Code

**Exact next task**

LEARNING session: Phase 3 start. What a neuron does — weights, bias, activation function. Then build a small network using PyTorch's `nn.Module`.

---

### Session 4 — Phase 2 Learning start

| Field              | Value         |
| ------------------ | ------------- |
| Date               | 29 March 2026 |
| Session type       | LEARNING      |
| Mode stayed clean? | Yes           |

**What was accomplished**

- Tensors, CPU vs GPU, `requires_grad`, autograd, computational graph
- Human independently changed equation from x*2 to x*3 and predicted gradient correctly

**Exact next task**

LEARNING session: Loss functions and the forward pass.

---

## Permanent decisions archive

| Decision                                     | Made in session   | Reason — never change because...                                                                                                                                                                                                                                                       |
| -------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Build on local PC only — no cloud            | Session 0 (setup) | The goal is to understand every part of the system. Cloud abstracts too much.                                                                                                                                                                                                          |
| Language: Python + PyTorch                   | Session 0 (setup) | Industry standard. Best learning resources. GPU support built in.                                                                                                                                                                                                                      |
| Use Python 3.11 venv, not system Python 3.14 | Session 1 (setup) | PyTorch has no CUDA build for Python 3.14. Always activate venv before any work. Command: C:\projects\myLLM\venv\Scripts\activate                                                                                                                                                      |
| Documents stored as .md files, not .docx     | Session 1 (setup) | Markdown is plain text — easier to read, edit, version, and paste into Claude.                                                                                                                                                                                                         |
| Use VS Code play button to run files         | Session 3         | Ensures the correct venv Python is used. Running python directly in terminal can pick up wrong interpreter.                                                                                                                                                                            |
| Human copies code, does not type it          | Session 5         | Comments are half the learning. Human confirmed they would skip comments if writing themselves. Copying is faster and preserves the full annotated explanation.                                                                                                                        |
| Matplotlib used for in-editor plots          | Session 5         | Human wanted to see training behaviour visually inside VS Code. Superseded by Session 8 decision below.                                                                                                                                                                                |
| GPU move deferred to Phase 6                 | Session 7         | Learning networks are too small to benefit from GPU. Superseded by Session 8 decision below.                                                                                                                                                                                           |
| No Matplotlib from Session 8 onward          | Session 8         | Human prefers print-only output — faster to run, less distraction. All loss tracking via print statements only.                                                                                                                                                                        |
| GPU enabled from Session 8 onward            | Session 8         | Phase 3 complete — all learning concepts done. Standard pattern every file must use: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, then `.to(device)` on inputs, labels, and model. Data and model must always be on the same device or PyTorch will error. |

---

_End of Document 3 — Session Log — v9_
