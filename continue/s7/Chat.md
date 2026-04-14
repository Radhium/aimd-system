# Session type: LEARNING

`Project phase`: Phase 3 — First neural network (final concept)

What was done last session (Session 7): You built network2.py — a two-input network that learned to discover y = 2x₁ + 3x₂. You discovered convergence yourself by running 20,000 epochs and observing the weights barely changed. You understood distributed representations — that no single neuron holds the full answer, it's spread across all of them.

Today's goal: Softmax and cross-entropy loss. This is the last concept in Phase 3. After this session, Phase 3 is complete.

What this session is about in plain terms: Everything so far has been regression — the network outputs a number (like 5.0039). Today you'll learn what happens when the output needs to be a category — like "is this the letter A, B, or C?" That requires two new tools: Softmax and cross-entropy loss.

## Ready to begin. Let's start with the core question.

**Why regression isn't enough**

Your current network outputs one number. If you ask it to predict y = 2x₁ + 3x₂, it gives back something like 5.003. That's fine — there's a real number sitting there in the world and the network is trying to match it.
Now imagine a different task: given a small set of numbers, predict whether they describe the letter A, B, or C.
What should the output be? You can't say "the answer is 2.7". Categories don't have a natural number. You need the network to say something like:
`"I'm 80% sure it's A, 15% sure it's B, 5% sure it's C."`
That's a probability distribution — three numbers that are all positive and all add up to 1.0. That's exactly what Softmax produces.

**Part 1 — Softmax**

Your network's last layer will output raw scores — one per category. These are called logits (we'll add this to the glossary). They're just numbers — could be negative, could be large, not probabilities yet.

Example logits for 3 categories: [2.0, 1.0, 0.1]

Softmax turns these into probabilities using this formula:

```For each score z_i:
softmax(z_i) = e^(z_i) / sum of all e^(z_j)
```

`e` is `Euler's number` — approximately 2.718. e^x is just a mathematical way of making every number positive and exaggerating the differences.

Let's walk through it with [2.0, 1.0, 0.1]:

```
Raw score|  e^z| Probability

2.0 |7.389| 7.389 / 10.337 = 71.6%
1.0 |2.718| 2.718 / 10.337 = 26.3%
0.1 |1.105| 1.105 / 10.337 = 10.7%
     Sum: 10.337|     Sum: ~100%

```

Three things Softmax guarantees:

1. All outputs are between 0 and 1
2. All outputs sum to exactly 1.0
3. The biggest raw score gets the biggest probability — the ranking is preserved

`Warm-up question — answer before reading on:`

```
If the raw scores are [3.0, 3.0, 3.0] — all equal — what do you expect Softmax to output? Think about it first, then tell me your answer.
```

(Answer this and I'll continue with cross-entropy loss — the second half of today's session.)

```
since all are 3, the e power z and sum of that with probability should be around 33% or so. so I expect softmax output should preserve all ranking same.
```

Exactly right on both counts.Since all three scores are equal, e^3 / (e^3 + e^3 + e^3) = e^3 / 3×e^3 = 1/3 — so each gets exactly 33.3%. And you correctly spotted the second point: Softmax preserves ranking, so when there's a tie in the inputs, there's a tie in the outputs.
This is actually a useful thing to keep in mind. If your network is uncertain — hasn't learned anything yet — all its outputs will be near-random, and Softmax will give roughly equal probabilities to every class. You can use that as a sanity check early in training.

---

`Part 2 — Cross-entropy loss`

Now you have probabilities. You need a loss function — something that measures how wrong the network was, so the gradient can fix it.

MSE won't work well here. MSE measures distance between numbers. But probabilities have a different structure — being 1% confident in the right answer is much worse than being 40% confident, and you want the loss to reflect that sharply.

Cross-entropy loss does exactly that. Here's the formula:

`Loss = -log(probability assigned to the correct class)`

That's it. You only look at what probability the network gave to the right answer. You ignore the rest.

Let's see what this looks like in practice. Suppose the correct answer is class A.

```
Network's probability for class A| Loss = -log(p)
0.99 (very confident, correct) |-log(0.99) = 0.01 — tiny loss
0.50 (uncertain)| -log(0.50) = 0.69
0.10 (mostly wrong)| -log(0.10) = 2.30
0.01 (very confident, wrong)| -log(0.01) = 4.60 — huge loss
```

The pattern: the more confident the network is in the wrong direction, the more it gets punished. The curve is not linear — being 1% on the right answer hurts far more than twice as much as being 10%. That sharp punishment is what makes classification training work.

---

**How Softmax and cross-entropy connect**

In PyTorch, these two are almost always used together, and PyTorch provides a single function that combines them efficiently:
`pythonnn.CrossEntropyLoss()`

This one function takes the raw logits (before Softmax) and the correct class index (just an integer — 0, 1, or 2), and internally applies Softmax + log + negation in one step. You never need to call Softmax yourself during training.

So the full picture for a classification network:

`Input → Hidden layers → Raw scores (logits) → CrossEntropyLoss ← correct class index`

During inference (when you want to actually see the probabilities), you apply Softmax manually to the logits. During training, nn.CrossEntropyLoss handles it.

Second warm-up question before we build the code:

Your network outputs logits [1.5, 0.2, 0.8] for classes A, B, C. The correct answer is B (index 1). Without calculating exactly — which class is the network most confident about right now, and is the loss going to be small or large?

```
1. class A is most confident right now.
2. the loss is going to be huge.
```

Both correct. Class A has the highest logit (1.5), so after Softmax it gets the biggest probability. But the correct answer is B, which has the lowest-ish score (0.2). The network is confidently wrong — so the loss will be large. Exactly the situation cross-entropy punishes hardest.

**Building the classification network**

Here is the code for today. Same structure as network2.py — but the task is now classification, not regression.
The task: Given two numbers, classify which of three categories they belong to. The network has to learn to separate them.

Create a new file called network3.py in C:\projects\myLLM\ and paste this in:

```python
# network3.py

'''
# Phase 3 — Classification with Softmax + Cross-Entropy Loss

#

# Task: given two input numbers, predict which of 3 categories they belong to.

# This is the first time the network outputs a category instead of a number.

#

# New concepts used here:

# - nn.CrossEntropyLoss : combines Softmax + log + negation in one step

# - Class labels : integers (0, 1, 2) instead of continuous targets

# - torch.argmax : picks the index of the highest value — the predicted class

# - Accuracy : how often the predicted class matches the true class
'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── 1. Training data ──────────────────────────────────────────────────────────

#

# 9 examples. Each row is [x1, x2]. The label is which category it belongs to.

#

# Category 0: inputs are small (around 0.1)

# Category 1: inputs are medium (around 0.5)

# Category 2: inputs are large (around 0.9)

#

# The network does not know these rules — it has to discover them from the data.

inputs = torch.tensor([
[0.1, 0.1],
[0.2, 0.1],
[0.1, 0.2],
[0.5, 0.5],
[0.6, 0.4],
[0.4, 0.6],
[0.9, 0.9],
[0.8, 0.9],
[0.9, 0.8],
], dtype=torch.float32)

# Labels are integers — 0, 1, or 2. Not floats. CrossEntropyLoss expects this.

labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

# ── 2. Model ──────────────────────────────────────────────────────────────────

#

# Input: 2 features (x1, x2)

# Hidden layer: 8 neurons with ReLU

# Output layer: 3 neurons — one raw score (logit) per category

#

# No Softmax here — CrossEntropyLoss applies it internally during training.

class Classifier(nn.Module):
def **init**(self):
super().**init**()
self.hidden = nn.Linear(2, 8) # 2 inputs → 8 hidden neurons
self.relu = nn.ReLU()
self.output = nn.Linear(8, 3) # 8 hidden → 3 output scores

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x   # raw logits — not probabilities yet

model = Classifier()

# ── 3. Loss and optimiser ─────────────────────────────────────────────────────

#

# CrossEntropyLoss takes (raw logits, integer labels).

# It internally applies Softmax, then computes -log(correct class probability).

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.05)

# ── 4. Training loop ──────────────────────────────────────────────────────────

loss_history = []

for epoch in range(500):
optimiser.zero_grad()

    logits = model(inputs)          # forward pass — shape: [9, 3]
    loss   = loss_fn(logits, labels)  # compare logits to correct class indices

    loss.backward()
    optimiser.step()

    loss_history.append(loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch:>4} | Loss: {loss.item():.4f}")

# ── 5. Final accuracy ─────────────────────────────────────────────────────────

#

# torch.argmax picks the index of the highest logit — the predicted class.

# We compare predictions to true labels and count how many match.

with torch.no_grad():
logits = model(inputs)
predictions = torch.argmax(logits, dim=1) # shape: [9] — one class per example
correct = (predictions == labels).sum().item()
accuracy = correct / len(labels) \* 100
print(f"\nFinal accuracy: {correct}/{len(labels)} = {accuracy:.1f}%")
print(f"Predicted: {predictions.tolist()}")
print(f"True: {labels.tolist()}")

# ── 6. Plot ───────────────────────────────────────────────────────────────────

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Classification training — loss over time")
plt.grid(True)
plt.tight_layout()
plt.show()
```

Run it and tell me:

1. What does the final accuracy print?
2. What do the predicted and true label lists look like — do they match?
3. What does the loss curve look like — same fast-drop-then-tail shape as before, or different?

```Bash
user@Shree-Ji-PC MINGW64 /c/projects/myLLM
$ c:/projects/myLLM/venv/Scripts/python.exe c:/projects/myLLM/network3.py
Epoch 0 | Loss: 1.0877
Epoch 50 | Loss: 0.0142
Epoch 100 | Loss: 0.0038
Epoch 150 | Loss: 0.0021
Epoch 200 | Loss: 0.0013
Epoch 250 | Loss: 0.0009
Epoch 300 | Loss: 0.0007
Epoch 350 | Loss: 0.0005
Epoch 400 | Loss: 0.0004
Epoch 450 | Loss: 0.0003
Final accuracy: 9/9 = 100.0%
Predicted: [0, 0, 0, 1, 1, 1, 2, 2, 2]
True: [0, 0, 0, 1, 1, 1, 2, 2, 2]

make two decisions...
one is from now on no need to matplotlib, meaning no visualization needed.
two is write the code, so the code runs on gpu not cpu.Show more
```

Good results first — then your two decisions.

`The output is perfect.` 9/9 accuracy, predictions match labels exactly. Loss dropped from 1.09 to 0.0003. You'll notice it dropped faster than before — by epoch 50 it was already at 0.014. That's because the learning rate is 0.05 here vs the smaller rates in earlier networks, and the task has clean separation between categories.

The shape is the same though: steep early drop, then a long slow tail as the network fine-tunes confidence on examples it already gets right.

Your two decisions

`Decision 1 — No more Matplotlib.` Noted and logged. From now on, code will print loss at intervals but won't generate plots. Faster to run, less distraction.

`Decision 2 — Run on GPU from now on.` Good timing. The networks so far were too small to benefit, but you've now seen all the concepts. Let's add GPU support properly.
Here is what changes — it's exactly four lines, always the same pattern:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs = inputs.to(device)
labels = labels.to(device)
model = model.to(device)

```

That's it. The training loop and loss function don't change — PyTorch handles everything else automatically once the data and model are on the same device.

One rule to remember: `data and model must always be on the same device.` If inputs are on GPU and the model is on CPU, PyTorch will throw an error. The pattern above guarantees they move together.

Both decisions are logged in the permanent decisions archive below. Phase 3 is now complete.
