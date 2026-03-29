# DOC 1 — Concept Map

**v4 · What has been learned · Updated every Learning session**

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

| Concept                               | Status      | Human's own words (Claude fills)                                                                                                                                                                                                             | Gaps / revisit                                                   |
| ------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| NumPy arrays and operations           | CAN EXPLAIN | "A vector is a list of items, a matrix is a grid, a tensor is a stack of grids. Shape is about structure not units." Predicted shapes correctly, broke matrix multiply intentionally, fixed it by reasoning backwards from the result shape. | None — practised in code                                         |
| Matrix multiplication                 | CAN EXPLAIN | "The middle number must match. The outer numbers become the result shape." Reasoned backwards from a target shape `(2,4)` to figure out what `b` needed to be — derived it, didn't guess.                                                    | None — practised in code                                         |
| What a derivative means (intuitively) | UNDERSTOOD  | "How much of a difference in one number effects the other number." Clean and correct on first attempt.                                                                                                                                       | Formal notation not yet introduced                               |
| The chain rule                        | UNDERSTOOD  | "It's like doing the same thing as a derivative but step by step, for safety — because for large parameters it becomes a headache to solve directly. Each step only needs to know about its immediate neighbours."                           | Not yet practised in code — will appear naturally in backprop    |
| What a gradient is                    | UNDERSTOOD  | Understood as a collection of derivatives — one per weight — packaged as a vector. Grasped the distinction from a single derivative cleanly.                                                                                                 | Gradient descent only introduced conceptually — not practised    |
| Python functions, loops, classes      | UNDERSTOOD  | "self is a reference — it's how the object points back to itself and holds memory between calls. Without it, data evaporates the moment the function ends. Multiple objects each have their own self so they don't interfere."               | Not yet used in project code — will appear when we define layers |

---

## Phase 2 — PyTorch fundamentals

| Concept                          | Status      | Human's own words (Claude fills)                                                                                                                                | Gaps / revisit                    |
| -------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| What a tensor is                 | NOT STARTED | [ Claude fills ]                                                                                                                                                |                                   |
| Autograd and computational graph | NOT STARTED | [ Claude fills ]                                                                                                                                                |                                   |
| Forward pass                     | NOT STARTED | [ Claude fills ]                                                                                                                                                |                                   |
| Loss function                    | INTRODUCED  | "A single number measuring how wrong the network's prediction was. Lower is better." Introduced briefly when human asked what "how wrong the network is" meant. | Full session on loss not yet done |
| Backward pass (backpropagation)  | NOT STARTED | [ Claude fills ]                                                                                                                                                |                                   |
| Optimizer (SGD, Adam)            | NOT STARTED | [ Claude fills ]                                                                                                                                                |                                   |
| Training loop structure          | NOT STARTED | [ Claude fills ]                                                                                                                                                |                                   |

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

| Concept               | The analogy that worked                                                                                                                                                | Session it clicked |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| Derivative            | Speedometer — your total distance is y, your speed right now is the derivative. It changes constantly.                                                                 | Session 2          |
| Gradient              | Standing on an open hillside in fog — the gradient is an arrow pointing in the direction of steepest climb. Walk the opposite way to go downhill.                      | Session 2          |
| Matrix multiplication | Shop bill — customers × quantities paired with prices × 1. Middle number is the shared thing (products). If it doesn't match, a price has nothing to multiply against. | Session 2          |
| Chain rule            | Oven → browning → timing → moisture. Each step only knows its neighbours. Multiply the effects as you walk back through the chain.                                     | Session 3          |
| self in a class       | A name tag each object wears. `self.count` means "the count that belongs to this specific object." Without it, data evaporates after each function call.               | Session 3          |

---

## Glossary — terms already explained

| Term                       | Plain English definition                                                                                                      | First explained in session |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| Virtual environment (venv) | A self-contained Python installation for one project — keeps packages isolated so they don't conflict with other projects     | Setup session              |
| CUDA                       | NVIDIA's system that lets PyTorch run computations on the GPU instead of the CPU — much faster for matrix math                | Setup session              |
| VRAM                       | Video RAM — the memory on your GPU. Limits how large your model and batch size can be                                         | Setup session              |
| Array                      | A collection of numbers arranged in a shape — 1D (vector), 2D (matrix), or 3D+ (tensor). Shape is about structure, not units. | Session 2                  |
| Vector                     | A 1D array — a single line of numbers. Shape shown as `(n,)`                                                                  | Session 2                  |
| Matrix                     | A 2D array — a grid of numbers with rows and columns. Shape shown as `(rows, cols)`                                           | Session 2                  |
| Tensor                     | A 3D (or higher) array — a stack of grids                                                                                     | Session 2                  |
| Derivative                 | A number that tells you how much changing one input changes the output — the slope at one specific point                      | Session 2                  |
| Gradient                   | A collection of derivatives — one per weight in the network — packaged as a vector                                            | Session 2                  |
| Gradient descent           | The process of walking opposite to the gradient to reduce loss — how networks learn                                           | Session 2                  |
| Loss                       | A single number measuring how wrong the network's prediction was. Lower is better.                                            | Session 2                  |
| Weight                     | A number inside a neural network that gets adjusted during training                                                           | Session 2                  |
| Chain rule                 | How to find the derivative of a chain of connected steps — multiply the derivative at each step as you walk backwards         | Session 3                  |
| Function                   | A named machine — takes input, does something, returns output. Defined with `def`.                                            | Session 3                  |
| Loop                       | A way to repeat something — `for` loops over a sequence, running the indented code once per item                              | Session 3                  |
| Class                      | A blueprint for building objects. Each object holds its own data (`self`) and has functions attached to it.                   | Session 3                  |
| self                       | A reference an object uses to point back to itself — how it stores and accesses its own data between function calls           | Session 3                  |
| `__init__`                 | The setup function that runs automatically when an object is created from a class                                             | Session 3                  |
| np.matmul                  | NumPy's matrix multiplication function — requires middle dimensions to match                                                  | Session 3                  |

---

_End of Document 1 — Concept Map — v4_
