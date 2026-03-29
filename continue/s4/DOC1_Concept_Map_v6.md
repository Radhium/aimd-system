# DOC 1 — Concept Map

**v7 · What has been learned · Updated every Learning session**

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

| Concept                          | Status      | Human's own words (Claude fills)                                                                                                                                                                                                                                                                                                                          | Gaps / revisit                                                                                              |
| -------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| What a tensor is                 | CAN EXPLAIN | "Like a NumPy array but with two superpowers — it can live on the GPU and it can remember operations." Predicted shapes correctly first attempt. Verified GPU/CPU device behaviour by running code. Asked unprompted why it says `cuda:0` instead of `gpu` — understood the index explanation immediately.                                                | None                                                                                                        |
| Autograd and computational graph | UNDERSTOOD  | "PyTorch watches the operations and writes them down like a receipt. `.backward()` reads the receipt backwards and gives you the gradient." Correctly predicted gradient would be `tensor([3.])` when equation changed from `x * 2` to `x * 3` — verified by running it independently.                                                                    | Not yet tested on a multi-step chain with more complex equations — will arise naturally in backprop session |
| Forward pass                     | UNDERSTOOD  | Understood as data moving in one direction — input through the network, out comes a prediction. No learning happens during this step. Correctly read the code and understood that `weight × input` is the simplest possible forward pass.                                                                                                                 | Not yet practised on a multi-layer network — will deepen in Phase 3                                         |
| Loss function (MSE)              | UNDERSTOOD  | Correctly predicted loss = 0 when weight = 3.0 by working through the arithmetic independently without running code. Understood why squaring is used — removes sign, punishes large errors harder. Key distinction grasped: input vs weight — misconception surfaced and corrected cleanly.                                                               | MSE only — cross-entropy loss comes in Phase 5                                                              |
| Backward pass                    | UNDERSTOOD  | Gradient came out negative — understood immediately that subtracting a negative means the weight goes up. Correctly predicted direction of weight change before running code. Verified: weight moved from 0.5 → 1.3, loss dropped from 100 → 46.24 in one step.                                                                                           | Not yet seen on a multi-weight network — will deepen in Phase 3                                             |
| Training loop                    | UNDERSTOOD  | Ran 100 steps and watched weight climb from 0.5 to 3.0 and loss fall to 0. Independently changed print interval to see all 100 steps — noticed loss showed 0.0000 at step 25 while weight was still 2.9999, correctly reasoned the loss was not truly zero but below display precision. Added Matplotlib plot — confirmed two charts rendered in VS Code. | Loop currently has one weight and one input — will scale up in Phase 3                                      |
| Optimizer (SGD, Adam)            | NOT STARTED | [ Claude fills ]                                                                                                                                                                                                                                                                                                                                          |                                                                                                             |

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

| Concept                     | The analogy that worked                                                                                                                                                                                                                      | Session it clicked |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| Derivative                  | Speedometer — your total distance is y, your speed right now is the derivative. It changes constantly.                                                                                                                                       | Session 2          |
| Gradient                    | Standing on an open hillside in fog — the gradient is an arrow pointing in the direction of steepest climb. Walk the opposite way to go downhill.                                                                                            | Session 2          |
| Matrix multiplication       | Shop bill — customers × quantities paired with prices × 1. Middle number is the shared thing (products). If it doesn't match, a price has nothing to multiply against.                                                                       | Session 2          |
| Chain rule                  | Oven → browning → timing → moisture. Each step only knows its neighbours. Multiply the effects as you walk back through the chain.                                                                                                           | Session 3          |
| self in a class             | A name tag each object wears. `self.count` means "the count that belongs to this specific object." Without it, data evaporates after each function call.                                                                                     | Session 3          |
| Computational graph         | A shop receipt — shows every item and price, not just the total. Backpropagation reads it backwards to find where error came from.                                                                                                           | Session 4          |
| CPU vs GPU                  | Your desk vs a workbench across the room. Faster for big jobs but you have to carry the work over yourself — PyTorch never moves it without being asked.                                                                                     | Session 4          |
| Input vs weight             | The photo vs your intuition in a guessing game. The photo changes every round (input). Your internal mental model improves over time (weight). The weight starts naive — 0.5 is not meaningful, just a starting point that is clearly wrong. | Session 5          |
| Gradient sign and direction | Gradient is negative → subtracting a negative → weight goes up. Direction is not chosen — it falls out of the arithmetic automatically.                                                                                                      | Session 5          |

---

## Glossary — terms already explained

| Term                       | Plain English definition                                                                                                                                                    | First explained in session     |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| Virtual environment (venv) | A self-contained Python installation for one project — keeps packages isolated so they don't conflict with other projects                                                   | Setup session                  |
| CUDA                       | NVIDIA's system that lets PyTorch run computations on the GPU instead of the CPU — much faster for matrix math                                                              | Setup session                  |
| VRAM                       | Video RAM — the memory on your GPU. Limits how large your model and batch size can be                                                                                       | Setup session                  |
| Array                      | A collection of numbers arranged in a shape — 1D (vector), 2D (matrix), or 3D+ (tensor). Shape is about structure, not units.                                               | Session 2                      |
| Vector                     | A 1D array — a single line of numbers. Shape shown as `(n,)`                                                                                                                | Session 2                      |
| Matrix                     | A 2D array — a grid of numbers with rows and columns. Shape shown as `(rows, cols)`                                                                                         | Session 2                      |
| Tensor                     | A 3D (or higher) array — a stack of grids                                                                                                                                   | Session 2                      |
| Derivative                 | A number that tells you how much changing one input changes the output — the slope at one specific point                                                                    | Session 2                      |
| Gradient                   | A collection of derivatives — one per weight in the network — packaged as a vector                                                                                          | Session 2                      |
| Gradient descent           | The process of walking opposite to the gradient to reduce loss — how networks learn                                                                                         | Session 2                      |
| Loss                       | A single number measuring how wrong the network's prediction was. Lower is better.                                                                                          | Session 2                      |
| Weight                     | A number inside a neural network that gets adjusted during training. Starts at an arbitrary value — the network discovers the right value through training.                 | Session 2 (extended Session 5) |
| Chain rule                 | How to find the derivative of a chain of connected steps — multiply the derivative at each step as you walk backwards                                                       | Session 3                      |
| Function                   | A named machine — takes input, does something, returns output. Defined with `def`.                                                                                          | Session 3                      |
| Loop                       | A way to repeat something — `for` loops over a sequence, running the indented code once per item                                                                            | Session 3                      |
| Class                      | A blueprint for building objects. Each object holds its own data (`self`) and has functions attached to it.                                                                 | Session 3                      |
| self                       | A reference an object uses to point back to itself — how it stores and accesses its own data between function calls                                                         | Session 3                      |
| `__init__`                 | The setup function that runs automatically when an object is created from a class                                                                                           | Session 3                      |
| np.matmul                  | NumPy's matrix multiplication function — requires middle dimensions to match                                                                                                | Session 3                      |
| PyTorch tensor             | A NumPy-like array with two added superpowers: it can live on the GPU, and it can remember every operation performed on it (autograd)                                       | Session 4                      |
| `requires_grad`            | A flag you set on a tensor to tell PyTorch to watch it and record every operation that touches it                                                                           | Session 4                      |
| Computational graph        | PyTorch's internal receipt of every operation performed on a watched tensor — read backwards during backpropagation to compute gradients                                    | Session 4                      |
| `.backward()`              | The PyTorch call that reads the computational graph backwards and computes all gradients automatically                                                                      | Session 4                      |
| `.grad`                    | The attribute on a tensor that holds its computed gradient after `.backward()` has been called                                                                              | Session 4                      |
| `.to("cuda")`              | The PyTorch method that copies a tensor from CPU memory to GPU memory. The original stays where it was — always a copy, not a move.                                         | Session 4                      |
| `cuda:0`                   | NVIDIA's GPU system, device number zero. The `:0` is an index — if you have one GPU it is always zero. Same as Python list indexing.                                        | Session 4                      |
| Forward pass               | Data moving through the network in one direction — input in, prediction out. No learning happens during this step.                                                          | Session 5                      |
| Input                      | Data coming from outside the network. Changes every training example. The network receives it but does not own it.                                                          | Session 5                      |
| MSE (Mean Squared Error)   | A loss function: (prediction − target)². Squaring removes the sign and punishes large errors more than small ones.                                                          | Session 5                      |
| Target                     | The correct answer the network should have predicted. In training, comes from the dataset.                                                                                  | Session 5                      |
| Learning rate              | A small number that controls how large each weight update step is. Too large → overshoot. Too small → trains very slowly.                                                   | Session 5                      |
| Training step              | One full cycle: forward pass → loss → backward pass → weight update → zero gradient.                                                                                        | Session 5                      |
| `weight.grad.zero_()`      | Clears the gradient on a weight before the next training step. Required because PyTorch accumulates gradients by default.                                                   | Session 5                      |
| `torch.no_grad()`          | A context block that tells PyTorch not to record the operations inside it. Used during weight updates — we don't want the update itself tracked in the computational graph. | Session 5                      |
| Matplotlib                 | A Python library for drawing charts. `plt.show()` opens a window with the plot after training finishes.                                                                     | Session 5                      |
| Divergence                 | When the learning rate is too large, the weight overshoots the target, overcorrects, and never settles. The loss explodes instead of falling.                               | Session 5                      |

---

_End of Document 1 — Concept Map — v7_
