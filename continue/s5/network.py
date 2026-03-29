# network.py
# Phase 3 — First real neural network using nn.Module
# Task: learn to map input x to target y = 3x
# Same task as Phase 2, but now using PyTorch's proper building blocks

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── 1. Define the network ──────────────────────────────────────────────────────

class SimpleNetwork(nn.Module):
    # nn.Module is the base class for all PyTorch models.
    # Inheriting from it means PyTorch automatically tracks all weights and biases
    # declared inside this class — we don't manage them by hand.

    def __init__(self):
        # __init__ runs once when we create the network.
        # We declare the layers here — not the data, just the structure.

        super().__init__()
        # super().__init__() activates nn.Module's internal bookkeeping.
        # Always call this first — without it, PyTorch won't track anything.

        self.layer1 = nn.Linear(in_features=1, out_features=8)
        # Linear layer: 1 input → 8 neurons.
        # PyTorch creates 8 weights and 8 biases automatically.
        # All are initialised to small random values.

        self.layer2 = nn.Linear(in_features=8, out_features=1)
        # Linear layer: 8 inputs → 1 output neuron.
        # Takes the 8 outputs from layer1 and combines them into one prediction.

        self.relu = nn.ReLU()
        # ReLU activation function — applied after layer1.
        # max(0, z): passes positive values, blocks negatives.
        # Without this, stacking layers adds no power (as explained in the session).

    def forward(self, x):
        # forward() describes how data flows through the network.
        # PyTorch calls this automatically when you pass data into the model.
        # x is the input — one number, but shaped as a tensor.

        z = self.layer1(x)
        # Pass input through layer1.
        # Each of the 8 neurons computes: (weight × input) + bias
        # Result shape: 8 numbers

        z = self.relu(z)
        # Apply ReLU to all 8 outputs.
        # Negative values become 0. Positive values pass through unchanged.

        z = self.layer2(z)
        # Pass the 8 ReLU outputs through layer2.
        # Combines them into a single prediction number.

        return z
        # Return the final prediction.

# ── 2. Create the network and the optimiser ────────────────────────────────────

model = SimpleNetwork()
# Creates one instance of the network.
# All weights and biases exist now, set to small random values.

print("Parameters inside the model:")
for name, param in model.named_parameters():
    print(f"  {name}: shape {param.shape}")
# named_parameters() walks through every learnable number in the model.
# This is the bookkeeping nn.Module does for you automatically.
# You'll see layer1.weight, layer1.bias, layer2.weight, layer2.bias.

print()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# The optimiser updates all weights and biases after each backward pass.
# Adam is a smarter version of SGD — it adjusts the learning rate per parameter
# automatically, so it tends to train faster and more stably than plain SGD.
# model.parameters() hands Adam every learnable number in the network.
# lr=0.01 is the learning rate — controls step size.

loss_fn = nn.MSELoss()
# PyTorch's built-in MSE loss function.
# Does exactly what your manual (prediction - target)**2 did in Phase 2,
# but handles batches cleanly and is ready for more complex use later.

# ── 3. Training data ───────────────────────────────────────────────────────────

# Same task as Phase 2: learn that output = 3 × input.
# We generate 100 input/target pairs spread across a range.

x_train = torch.linspace(-2, 2, 100).unsqueeze(1)
# linspace: 100 evenly spaced numbers from -2 to 2.
# unsqueeze(1): reshapes from shape (100,) to shape (100, 1).
# nn.Linear expects each input to be a row — shape (batch_size, features).

y_train = 3 * x_train
# Targets: the correct answer for each input is 3× the input.
# The network doesn't know this rule — it has to discover it from the data.

# ── 4. Training loop ───────────────────────────────────────────────────────────

epochs = 500
# One epoch = one full pass through all 100 training examples.
# 500 epochs gives the network enough steps to learn the pattern well.

loss_history = []

for epoch in range(epochs):

    # Forward pass — run all 100 inputs through the network
    predictions = model(x_train)
    # Calling model(...) automatically calls model.forward(...).
    # predictions shape: (100, 1) — one prediction per input.

    # Compute loss
    loss = loss_fn(predictions, y_train)
    # Compares all 100 predictions to all 100 targets.
    # Returns one number: the average squared error across the batch.

    # Backward pass
    optimizer.zero_grad()
    # Clears gradients from the previous step.
    # Same reason as Phase 2: PyTorch accumulates gradients by default.
    # Must do this before every backward pass.

    loss.backward()
    # Computes gradients for every weight and bias in the network.
    # The chain rule runs automatically through layer2 → relu → layer1.

    optimizer.step()
    # Adam reads all the gradients and updates every weight and bias.
    # One line replaces the manual weight update from Phase 2.

    loss_history.append(loss.item())
    # .item() converts the loss tensor to a plain Python float for storing.

    if epoch % 50 == 0:
        print(f"Epoch {epoch:>4} | Loss: {loss.item():.6f}")

# ── 5. Test the trained network ────────────────────────────────────────────────

print("\nTesting trained network:")
test_inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]

for val in test_inputs:
    x = torch.tensor([[val]])
    # Shape must be (1, 1) — one example, one feature.

    with torch.no_grad():
        # Disable gradient tracking during inference.
        # We are not training here — no need to build the computational graph.
        pred = model(x)

    print(f"  input: {val:>5.1f} | predicted: {pred.item():>7.4f} | target: {3*val:>7.4f}")

# ── 6. Plot loss curve ─────────────────────────────────────────────────────────

plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.title("Training Loss — SimpleNetwork")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()