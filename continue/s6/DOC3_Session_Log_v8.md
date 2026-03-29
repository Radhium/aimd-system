# DOC 3 — Session Log

**v9 · Claude's memory across sessions**

_Claude fills this at the end of every session. The human saves it and uploads it at the start of every future session. This document is the only continuity between chats._

Keep the last 5 session entries. Move critical decisions to the Permanent Decisions section at the bottom. Delete older entries but never delete the Permanent Decisions.

---

## Project snapshot

| Field               | Value                                                                                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Project name        | My LLM from scratch                                                                                                                                                       |
| Current phase       | Phase 3 — First neural network (in progress)                                                                                                                              |
| Overall progress    | 35% — multiple input features understood, distributed representations understood, convergence understood. One concept remaining in Phase 3: Softmax + cross-entropy loss. |
| Last session date   | 29 March 2026                                                                                                                                                             |
| Last session type   | LEARNING                                                                                                                                                                  |
| Next session type   | LEARNING — Phase 3 final concept                                                                                                                                          |
| Next immediate task | Softmax and cross-entropy loss — what happens when the output is a category (not a number). Build a small classification network. Phase 3 complete after this.            |

---

## Phase and concept progress

| Phase                              | Status      | Notes                                                                                                                                                                                                                                      |
| ---------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Phase 1 — Python + math            | COMPLETE    | All core concepts understood. NumPy practised in code. Chain rule, functions, loops, classes done.                                                                                                                                         |
| Phase 2 — PyTorch fundamentals     | COMPLETE    | Tensors, autograd, forward pass, MSE loss, backward pass, training loop all covered and practised in code. Matplotlib plots added. Optimizer (SGD/Adam) not formally covered — will introduce naturally in Phase 3 when nn.Module is used. |
| Phase 3 — First neural network     | IN PROGRESS | Neuron, nn.Module, nn.Linear, Adam, two-layer network, multiple input features, hidden layer representations, convergence all understood. Remaining: Softmax + cross-entropy loss (one session).                                           |
| Phase 4 — Transformer architecture | NOT STARTED |                                                                                                                                                                                                                                            |
| Phase 5 — Dataset + pipeline       | NOT STARTED |                                                                                                                                                                                                                                            |
| Phase 6 — Train the model          | NOT STARTED |                                                                                                                                                                                                                                            |
| Phase 7 — Serve it locally         | NOT STARTED |                                                                                                                                                                                                                                            |

---

## Session entries

Most recent at the top. Keep last 5 entries only.

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
- Noted GPU not used — explained PyTorch defaults to CPU; will add `.to("cuda")` properly in Phase 6

**Concepts covered**

- Multiple input features / weighted sum — UNDERSTOOD
- What a hidden layer learns (distributed representations) — UNDERSTOOD
- Convergence (empirically discovered) — UNDERSTOOD
- Output layer as weighted combination of hidden signals — UNDERSTOOD
- GPU vs CPU default in PyTorch — UNDERSTOOD

**What is unfinished or unclear**

- Softmax and cross-entropy loss not yet covered — next session
- GPU integration deferred to Phase 6

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

### Session 3 — Phase 1 Learning continued

| Field              | Value         |
| ------------------ | ------------- |
| Date               | 28 March 2026 |
| Session type       | LEARNING      |
| Mode stayed clean? | Yes           |

**What was accomplished**

- Chain rule, Python functions/loops/classes, NumPy exercises
- Moved from terminal to VS Code — fixed wrong Python interpreter

**Exact next task**

LEARNING session: Phase 2 start. PyTorch tensors and autograd.

---

## Permanent decisions archive

| Decision                                     | Made in session   | Reason — never change because...                                                                                                                                |
| -------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Build on local PC only — no cloud            | Session 0 (setup) | The goal is to understand every part of the system. Cloud abstracts too much.                                                                                   |
| Language: Python + PyTorch                   | Session 0 (setup) | Industry standard. Best learning resources. GPU support built in.                                                                                               |
| Use Python 3.11 venv, not system Python 3.14 | Session 1 (setup) | PyTorch has no CUDA build for Python 3.14. Always activate venv before any work. Command: C:\projects\myLLM\venv\Scripts\activate                               |
| Documents stored as .md files, not .docx     | Session 1 (setup) | Markdown is plain text — easier to read, edit, version, and paste into Claude.                                                                                  |
| Use VS Code play button to run files         | Session 3         | Ensures the correct venv Python is used. Running python directly in terminal can pick up wrong interpreter.                                                     |
| Human copies code, does not type it          | Session 5         | Comments are half the learning. Human confirmed they would skip comments if writing themselves. Copying is faster and preserves the full annotated explanation. |
| Matplotlib used for in-editor plots          | Session 5         | Human wants to see training behaviour visually inside VS Code. Matplotlib pops a window after training. Installed in venv.                                      |
| GPU move deferred to Phase 6                 | Session 7         | Learning networks are too small to benefit from GPU. Add `.to("cuda")` properly when writing the real training loop.                                            |

---

_End of Document 3 — Session Log — v9_
