# DOC 3 — Session Log
**v3 · Claude's memory across sessions**

*Claude fills this at the end of every session. The human saves it and uploads it at the start of every future session. This document is the only continuity between chats.*

Keep the last 5 session entries. Move critical decisions to the Permanent Decisions section at the bottom. Delete older entries but never delete the Permanent Decisions.

---

## Project snapshot

| Field | Value |
|---|---|
| Project name | My LLM from scratch |
| Current phase | Phase 1 — Python and math foundations (IN PROGRESS) |
| Overall progress | 5% — environment ready, first learning session done |
| Last session date | 28 March 2026 |
| Last session type | LEARNING |
| Next session type | LEARNING — continue Phase 1 |
| Next immediate task | The chain rule — how derivatives chain together. Then Python functions, loops, and classes. Then NumPy and matrix multiplication in actual code. |

---

## Phase and concept progress

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — Python + math | IN PROGRESS | Arrays, matrix multiply, derivative, gradient understood. Chain rule and Python basics still to do. |
| Phase 2 — PyTorch fundamentals | NOT STARTED | Loss introduced briefly |
| Phase 3 — First neural network | NOT STARTED | |
| Phase 4 — Transformer architecture | NOT STARTED | |
| Phase 5 — Dataset + pipeline | NOT STARTED | |
| Phase 6 — Train the model | NOT STARTED | |
| Phase 7 — Serve it locally | NOT STARTED | |

---

## Session entries

Most recent at the top. Keep last 5 entries only.

---

### Session 2 — Phase 1 Learning start

| Field | Value |
|---|---|
| Date | 28 March 2026 |
| Session type | LEARNING |
| Mode stayed clean? | Yes |

**What was accomplished**

- Introduced NumPy arrays — 1D (vector), 2D (matrix), 3D (tensor)
- Corrected initial misconception: shape is about structure, not units
- Explained matrix multiplication using shop/bill analogy — middle numbers must match because they represent the shared thing being paired
- Explained derivative intuitively — how much changing one number affects another
- Explained gradient — collection of derivatives, one per weight, packaged as a vector
- Introduced loss briefly when human asked what "how wrong the network is" meant
- Human asked a sharp clarifying question unprompted — good sign

**Concepts covered**

- NumPy arrays and shape — UNDERSTOOD
- Matrix multiplication — UNDERSTOOD
- Derivative (intuitive) — UNDERSTOOD
- Gradient — UNDERSTOOD
- Loss — INTRODUCED (not full session yet)

**What is unfinished or unclear**

- Chain rule not yet covered
- Python functions, loops, classes not yet covered
- No NumPy code written yet — all conceptual so far
- Gradient descent only introduced conceptually

**Exact next task**

LEARNING session: The chain rule — how derivatives chain together through layers. Then Python functions and loops. Then write actual NumPy code for arrays and matrix multiplication.

---

### Session 1 — Setup

| Field | Value |
|---|---|
| Date | 28 March 2026 |
| Session type | SETUP (system improvement — not a standard session type) |
| Mode stayed clean? | Yes |

**What was accomplished**

- Read and understood all four system documents (Doc 0–3)
- Identified six gaps in the original document system
- Filled hardware environment table in Doc 2 using systeminfo and nvidia-smi
- Diagnosed that PyTorch was installed as CPU-only due to Python 3.14 incompatibility
- Installed Python 3.11.9 via py launcher alongside existing Python 3.14
- Created virtual environment at C:\projects\myLLM\venv using Python 3.11
- Reinstalled PyTorch 2.5.1+cu121 — GPU confirmed working (RTX 3050 visible to PyTorch)
- Upgraded all four documents from .docx to .md and applied all six gap fixes

**Concepts covered or code written**

- No learning concepts covered — this was a setup session
- No code written — environment configuration only

**What is unfinished or unclear**

- Project folder structure not yet created beyond the venv
- Python version discrepancy noted: system has 3.14, project uses 3.11 venv. Must always activate venv before working.

**Exact next task**

Learning session: Phase 1 start. Introduce NumPy arrays and matrix operations. Use analogies and small exercises. Check understanding before moving on.

---

### Session N-2

| Field | Value |
|---|---|
| Date | [ Claude fills ] |
| Session type | [ LEARNING / BUILD / EXPERIMENT / DEBUG / REVIEW ] |
| Mode stayed clean? | [ Yes / No ] |

**What was accomplished**

*[ Claude fills ]*

**Concepts covered or code written**

*[ Claude fills ]*

**What is unfinished or unclear**

*[ Claude fills ]*

**Exact next task**

*[ Claude fills ]*

---

### Session N-3

| Field | Value |
|---|---|
| Date | [ Claude fills ] |
| Session type | [ LEARNING / BUILD / EXPERIMENT / DEBUG / REVIEW ] |
| Mode stayed clean? | [ Yes / No ] |

**What was accomplished**

*[ Claude fills ]*

**Concepts covered or code written**

*[ Claude fills ]*

**What is unfinished or unclear**

*[ Claude fills ]*

**Exact next task**

*[ Claude fills ]*

---

### Session N-4 — Oldest kept

| Field | Value |
|---|---|
| Date | [ Claude fills ] |
| Session type | [ LEARNING / BUILD / EXPERIMENT / DEBUG / REVIEW ] |
| Mode stayed clean? | [ Yes / No ] |

**What was accomplished**

*[ Claude fills ]*

**Concepts covered or code written**

*[ Claude fills ]*

**What is unfinished or unclear**

*[ Claude fills ]*

**Exact next task**

*[ Claude fills ]*

---

## Permanent decisions archive

| Decision | Made in session | Reason — never change because... |
|---|---|---|
| Build on local PC only — no cloud | Session 0 (setup) | The goal is to understand every part of the system. Cloud abstracts too much. |
| Language: Python + PyTorch | Session 0 (setup) | Industry standard. Best learning resources. GPU support built in. |
| Use Python 3.11 venv, not system Python 3.14 | Session 1 (setup) | PyTorch has no CUDA build for Python 3.14. Always activate venv before any work. Command: C:\projects\myLLM\venv\Scripts\activate |
| Documents stored as .md files, not .docx | Session 1 (setup) | Markdown is plain text — easier to read, edit, version, and paste into Claude. |

---

*End of Document 3 — Session Log — v3*
