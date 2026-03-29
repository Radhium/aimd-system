# DOC 1 — Concept Map

**v2 · What has been learned · Updated every Learning session**

_This document tracks every concept the human has studied. Claude updates it after every Learning session. It is the map of understanding — not a to-do list, but a record of what is known, what is fuzzy, and what still needs work._

---

## How to use this document

- Claude fills every entry after a Learning session
- Status values: NOT STARTED | INTRODUCED | UNDERSTOOD | CAN EXPLAIN | DEEP
- 'Can explain' means the human could teach it to someone else
- 'Deep' means the human understands the math behind it, not just the concept

Claude: when filling the 'Human's own words' column — write what the human actually said during the session, not a textbook definition. This makes it personal and sticky.

---

## Phase 1 — Python and math foundations

| Concept                               | Status      | Human's own words (Claude fills) | Gaps / revisit |
| ------------------------------------- | ----------- | -------------------------------- | -------------- |
| NumPy arrays and operations           | NOT STARTED | [ Claude fills after session ]   |                |
| Matrix multiplication                 | NOT STARTED | [ Claude fills after session ]   |                |
| What a derivative means (intuitively) | NOT STARTED | [ Claude fills after session ]   |                |
| The chain rule                        | NOT STARTED | [ Claude fills after session ]   |                |
| What a gradient is                    | NOT STARTED | [ Claude fills after session ]   |                |
| Python functions, loops, classes      | NOT STARTED | [ Claude fills after session ]   |                |

---

## Phase 2 — PyTorch fundamentals

| Concept                          | Status      | Human's own words (Claude fills) | Gaps / revisit |
| -------------------------------- | ----------- | -------------------------------- | -------------- |
| What a tensor is                 | NOT STARTED | [ Claude fills ]                 |                |
| Autograd and computational graph | NOT STARTED | [ Claude fills ]                 |                |
| Forward pass                     | NOT STARTED | [ Claude fills ]                 |                |
| Loss function                    | NOT STARTED | [ Claude fills ]                 |                |
| Backward pass (backpropagation)  | NOT STARTED | [ Claude fills ]                 |                |
| Optimizer (SGD, Adam)            | NOT STARTED | [ Claude fills ]                 |                |
| Training loop structure          | NOT STARTED | [ Claude fills ]                 |                |

---

## Phase 3 — Neural network basics

| Concept                           | Status      | Human's own words (Claude fills) | Gaps / revisit |
| --------------------------------- | ----------- | -------------------------------- | -------------- |
| What a neuron does                | NOT STARTED | [ Claude fills ]                 |                |
| Activation functions (ReLU, etc.) | NOT STARTED | [ Claude fills ]                 |                |
| What a layer is                   | NOT STARTED | [ Claude fills ]                 |                |
| How a network learns              | NOT STARTED | [ Claude fills ]                 |                |
| Overfitting vs underfitting       | NOT STARTED | [ Claude fills ]                 |                |
| Train / validation / test split   | NOT STARTED | [ Claude fills ]                 |                |

---

## Phase 4 — The Transformer

| Concept                                   | Status      | Human's own words (Claude fills) | Gaps / revisit |
| ----------------------------------------- | ----------- | -------------------------------- | -------------- |
| Tokenization                              | NOT STARTED | [ Claude fills ]                 |                |
| Embeddings                                | NOT STARTED | [ Claude fills ]                 |                |
| Positional encoding                       | NOT STARTED | [ Claude fills ]                 |                |
| Self-attention mechanism                  | NOT STARTED | [ Claude fills ]                 |                |
| Multi-head attention                      | NOT STARTED | [ Claude fills ]                 |                |
| Query / Key / Value (Q, K, V)             | NOT STARTED | [ Claude fills ]                 |                |
| Feed-forward layers                       | NOT STARTED | [ Claude fills ]                 |                |
| Layer normalization                       | NOT STARTED | [ Claude fills ]                 |                |
| Residual connections                      | NOT STARTED | [ Claude fills ]                 |                |
| The decoder-only architecture (GPT style) | NOT STARTED | [ Claude fills ]                 |                |

---

## Phase 5 — Training a language model

| Concept                                              | Status      | Human's own words (Claude fills) | Gaps / revisit |
| ---------------------------------------------------- | ----------- | -------------------------------- | -------------- |
| Language modelling objective (next token prediction) | NOT STARTED | [ Claude fills ]                 |                |
| Cross-entropy loss                                   | NOT STARTED | [ Claude fills ]                 |                |
| Perplexity                                           | NOT STARTED | [ Claude fills ]                 |                |
| Learning rate schedules                              | NOT STARTED | [ Claude fills ]                 |                |
| Batch size and gradient accumulation                 | NOT STARTED | [ Claude fills ]                 |                |
| Checkpointing                                        | NOT STARTED | [ Claude fills ]                 |                |
| Reading a loss curve                                 | NOT STARTED | [ Claude fills ]                 |                |

---

## Analogies that clicked

_Claude fills this section during Learning sessions. When an analogy clearly makes the human understand something — write it down here so it can be reused in future sessions._

| Concept            | The analogy that worked                                                             | Session it clicked |
| ------------------ | ----------------------------------------------------------------------------------- | ------------------ |
| [ e.g. Attention ] | [ e.g. Every word looks at every other word and decides how much to care about it ] | [ Session # ]      |

---

## Glossary — terms already explained

_Claude adds a term here the first time it is explained to the human. Before explaining any term in a future session, check this list. If it's here — reference it briefly, don't re-explain from scratch._

| Term                       | Plain English definition                                                                                                  | First explained in session |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| Virtual environment (venv) | A self-contained Python installation for one project — keeps packages isolated so they don't conflict with other projects | Setup session              |
| CUDA                       | NVIDIA's system that lets PyTorch run computations on the GPU instead of the CPU — much faster for matrix math            | Setup session              |
| VRAM                       | Video RAM — the memory on your GPU. Limits how large your model and batch size can be                                     | Setup session              |

---

_End of Document 1 — Concept Map — v2_
