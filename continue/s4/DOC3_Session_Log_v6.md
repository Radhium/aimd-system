# DOC 3 — Session Log

**v7 · Claude's memory across sessions**

_Claude fills this at the end of every session. The human saves it and uploads it at the start of every future session. This document is the only continuity between chats._

Keep the last 5 session entries. Move critical decisions to the Permanent Decisions section at the bottom. Delete older entries but never delete the Permanent Decisions.

---

## Project snapshot

| Field               | Value                                                                                                                                                          |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Project name        | My LLM from scratch                                                                                                                                            |
| Current phase       | Phase 2 — PyTorch fundamentals (COMPLETE)                                                                                                                      |
| Overall progress    | 25% — full training loop understood and run in code. Ready for Phase 3.                                                                                        |
| Last session date   | 29 March 2026                                                                                                                                                  |
| Last session type   | LEARNING                                                                                                                                                       |
| Next session type   | LEARNING — Phase 3 start                                                                                                                                       |
| Next immediate task | What a neuron does — introduce the single neuron, weights + bias + activation function. Then build a two-layer network from scratch using PyTorch's nn.Module. |

---

## Phase and concept progress

| Phase                              | Status      | Notes                                                                                                                                                                                                                                      |
| ---------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Phase 1 — Python + math            | COMPLETE    | All core concepts understood. NumPy practised in code. Chain rule, functions, loops, classes done.                                                                                                                                         |
| Phase 2 — PyTorch fundamentals     | COMPLETE    | Tensors, autograd, forward pass, MSE loss, backward pass, training loop all covered and practised in code. Matplotlib plots added. Optimizer (SGD/Adam) not formally covered — will introduce naturally in Phase 3 when nn.Module is used. |
| Phase 3 — First neural network     | NOT STARTED |                                                                                                                                                                                                                                            |
| Phase 4 — Transformer architecture | NOT STARTED |                                                                                                                                                                                                                                            |
| Phase 5 — Dataset + pipeline       | NOT STARTED |                                                                                                                                                                                                                                            |
| Phase 6 — Train the model          | NOT STARTED |                                                                                                                                                                                                                                            |
| Phase 7 — Serve it locally         | NOT STARTED |                                                                                                                                                                                                                                            |

---

## Session entries

Most recent at the top. Keep last 5 entries only.

---

### Session 5 — Forward pass, loss, backward pass, training loop

| Field              | Value         |
| ------------------ | ------------- |
| Date               | 29 March 2026 |
| Session type       | LEARNING      |
| Mode stayed clean? | Yes           |

**What was accomplished**

- Explained forward pass — data moves one direction, input to output, no learning during this step
- Explained MSE loss — (prediction − target)², squaring removes sign and punishes large errors harder
- Wrote first version of `forward_pass.py` — one weight, one input, one forward pass, one loss calculation
- Human correctly predicted loss = 100 before running. Confirmed output: Loss: 100.0000
- Key misconception surfaced and corrected: human thought weight 0.5 was the input data. Corrected with photo/intuition analogy — clicked immediately
- Human correctly solved warm-up question independently: if weight = 3.0, loss = 0. Worked through arithmetic without code.
- Extended `forward_pass.py` with backward pass — called `loss.backward()`, printed gradient
- Gradient came out negative — human correctly predicted weight would go up, loss would go down
- Verified: weight 0.5 → 1.3, loss 100 → 46.24 in one step
- Extended to full 100-step training loop — weight climbed from 0.5 to 3.0, loss fell to 0
- Human independently changed print interval to see all 100 steps — sharp observation: loss showed 0.0000 at step 25 while weight was 2.9999, correctly reasoned loss was below display precision not truly zero
- Human asked to see visuals inside VS Code — installed Matplotlib, added plot to `forward_pass.py`
- Two charts rendered correctly in VS Code: loss curve and weight curve with target line

**Concepts covered**

- Forward pass — UNDERSTOOD
- Loss function (MSE) — UNDERSTOOD
- Input vs weight distinction — UNDERSTOOD (key misconception corrected)
- Backward pass — UNDERSTOOD
- Training loop — UNDERSTOOD
- Matplotlib plotting — introduced and working

**What is unfinished or unclear**

- Optimizer (SGD, Adam) not yet formally introduced — manual weight update used throughout. Will introduce `torch.optim` naturally in Phase 3.
- Training loop currently has one weight and one input — will scale to many weights in Phase 3

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

- Explained why PyTorch needs its own tensor type — NumPy cannot run on GPU and has no memory of operations
- Ran tensor exercises in `tensors.py` — shapes predicted correctly first attempt (`(3,)` and `(2,2)`)
- Explained CPU vs GPU — tensors start on CPU by default, `.to("cuda")` copies to GPU
- Human asked unprompted why device shows `cuda:0` instead of `gpu` — explained indexing, clicked immediately
- Confirmed `.to("cuda")` makes a copy — original stays on CPU — verified by running code
- Explained `requires_grad=True` — opts a tensor into PyTorch's operation tracking
- Explained computational graph using receipt analogy — PyTorch writes down every operation, reads it backwards during backprop
- Ran autograd exercise — human predicted gradient of `z = x * 2 + 1` would be linear before running
- Output was `tensor([2.])` — explained why: `+1` contributes nothing to rate of change, `* 2` is the slope
- Human independently changed equation to `x * 3`, predicted `tensor([3.])`, ran it, confirmed correct

**Concepts covered**

- PyTorch tensors — CAN EXPLAIN
- Autograd and computational graph — UNDERSTOOD

**What is unfinished or unclear**

- Loss function, forward pass, backward pass, optimizer, training loop not yet covered

**Exact next task**

LEARNING session: Loss functions and the forward pass. What a forward pass is, how a prediction is made, how loss is calculated from it, and writing a simple example in code.

---

### Session 3 — Phase 1 Learning continued

| Field              | Value         |
| ------------------ | ------------- |
| Date               | 28 March 2026 |
| Session type       | LEARNING      |
| Mode stayed clean? | Yes           |

**What was accomplished**

- Explained the chain rule — human understood it as "derivative step by step, each step only knows its neighbours, multiply as you walk back"
- Explained Python functions, loops, classes — self as object memory clicked cleanly
- Moved from terminal to VS Code — debugged wrong Python being selected, now correctly using venv
- Ran NumPy exercises — all four predicted and verified correctly including breaking and fixing matrix multiply

**Concepts covered**

- Chain rule — UNDERSTOOD
- Python functions, loops, classes — UNDERSTOOD
- NumPy arrays and operations — CAN EXPLAIN
- Matrix multiplication — CAN EXPLAIN

**Exact next task**

LEARNING session: Phase 2 start. PyTorch tensors and autograd.

---

### Session 2 — Phase 1 Learning start

| Field              | Value         |
| ------------------ | ------------- |
| Date               | 28 March 2026 |
| Session type       | LEARNING      |
| Mode stayed clean? | Yes           |

**What was accomplished**

- Introduced NumPy arrays, matrix multiplication, derivative, gradient, loss (briefly)
- All conceptual — no code yet

**Concepts covered**

- NumPy arrays and shape — UNDERSTOOD
- Matrix multiplication — UNDERSTOOD
- Derivative (intuitive) — UNDERSTOOD
- Gradient — UNDERSTOOD
- Loss — INTRODUCED

**Exact next task**

LEARNING session: Chain rule, Python functions and loops, then write actual NumPy code.

---

### Session 1 — Setup

| Field              | Value         |
| ------------------ | ------------- |
| Date               | 28 March 2026 |
| Session type       | SETUP         |
| Mode stayed clean? | Yes           |

**What was accomplished**

- Diagnosed CPU-only PyTorch due to Python 3.14 incompatibility
- Installed Python 3.11.9, created venv, reinstalled PyTorch 2.5.1+cu121 with GPU confirmed
- Upgraded all documents from .docx to .md

**Exact next task**

LEARNING session: Phase 1 start. NumPy arrays and matrix operations.

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

---

_End of Document 3 — Session Log — v7_
