import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── 1. Generate training data ───────────────────────────────────────────────
# The rule we want the network to discover: y = 2*x1 + 3*x2
# We never tell the network this formula — it must learn it from examples.

torch.manual_seed(42)  # makes results reproducible

# 200 training examples, each with 2 input features
X = torch.rand(200, 2)  # shape [200, 2] — 200 rows, 2 columns (x1 and x2)

# True targets: apply the hidden rule
# X[:, 0] means "all rows, column 0" → that's x1
# X[:, 1] means "all rows, column 1" → that's x2
y = 2 * X[:, 0] + 3 * X[:, 1]  # shape [200]
y = y.unsqueeze(1)               # reshape to [200, 1] so it matches the output shape

# ── 2. Define the network ───────────────────────────────────────────────────
class TwoInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Hidden layer: takes 2 inputs, produces 8 signals
        self.hidden = nn.Linear(in_features=2, out_features=8)
        self.relu   = nn.ReLU()
        # Output layer: takes 8 signals, collapses to 1 prediction
        self.output = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        x = self.hidden(x)   # weighted sum across both inputs, for each of 8 neurons
        x = self.relu(x)     # apply activation — kill negatives
        x = self.output(x)   # combine 8 signals into final prediction
        return x

model = TwoInputNet()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# Expected: (2*8 + 8) + (8*1 + 1) = 24 + 9 = 33

# ── 3. Loss and optimizer ───────────────────────────────────────────────────
loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ── 4. Training loop ────────────────────────────────────────────────────────
losses = []

for epoch in range(1000):
    # Forward pass
    predictions = model(X)

    # Compute loss
    loss = loss_fn(predictions, y)
    losses.append(loss.item())

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# ── 5. Inspect what the hidden layer learned ────────────────────────────────
print("\n── Hidden layer weights (shape [8, 2]) ──")
print(model.hidden.weight.data)
# Each row is one neuron. Each row has 2 numbers: weight for x1, weight for x2.
# After training, rows should show combinations that reflect the 2:3 ratio.

print("\n── Output layer weights (shape [1, 8]) ──")
print(model.output.weight.data)
# These combine the 8 hidden signals into one final answer.

# ── 6. Test on a known input ─────────────────────────────────────────────────
# What does the network predict for x1=1.0, x2=1.0?
# The true answer: y = 2*1 + 3*1 = 5.0
test_input = torch.tensor([[1.0, 1.0]])
with torch.no_grad():
    prediction = model(test_input)
print(f"\nTest: x1=1.0, x2=1.0 → predicted: {prediction.item():.4f} | true: 5.0")

# ── 7. Plot ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training loss — two-input network")
plt.grid(True)
plt.tight_layout()
plt.show()