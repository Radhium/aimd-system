# network3.py
# Phase 3 — Classification with Softmax + Cross-Entropy Loss
#
# Task: given two input numbers, predict which of 3 categories they belong to.
# This is the first time the network outputs a category instead of a number.
#
# New concepts used here:
#   - nn.CrossEntropyLoss  : combines Softmax + log + negation in one step
#   - Class labels         : integers (0, 1, 2) instead of continuous targets
#   - torch.argmax         : picks the index of the highest value — the predicted class
#   - Accuracy             : how often the predicted class matches the true class

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── 1. Training data ──────────────────────────────────────────────────────────
#
# 9 examples. Each row is [x1, x2]. The label is which category it belongs to.
#
# Category 0: inputs are small  (around 0.1)
# Category 1: inputs are medium (around 0.5)
# Category 2: inputs are large  (around 0.9)
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
# Input:        2 features (x1, x2)
# Hidden layer: 8 neurons with ReLU
# Output layer: 3 neurons — one raw score (logit) per category
#
# No Softmax here — CrossEntropyLoss applies it internally during training.

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 8)   # 2 inputs → 8 hidden neurons
        self.relu   = nn.ReLU()
        self.output = nn.Linear(8, 3)   # 8 hidden → 3 output scores

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

loss_fn   = nn.CrossEntropyLoss()
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
    logits      = model(inputs)
    predictions = torch.argmax(logits, dim=1)   # shape: [9] — one class per example
    correct     = (predictions == labels).sum().item()
    accuracy    = correct / len(labels) * 100
    print(f"\nFinal accuracy: {correct}/{len(labels)} = {accuracy:.1f}%")
    print(f"Predicted: {predictions.tolist()}")
    print(f"True:      {labels.tolist()}")

# ── 6. Plot ───────────────────────────────────────────────────────────────────

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Classification training — loss over time")
plt.grid(True)
plt.tight_layout()
plt.show()