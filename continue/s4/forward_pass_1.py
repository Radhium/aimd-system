import torch

# ============================================================
# FORWARD PASS AND LOSS — the simplest possible example
#
# Goal: the network should learn to multiply any input by 3.
# Today we do NOT train it. We just:
#   1. Make a prediction (forward pass)
#   2. Measure how wrong it was (loss)
# ============================================================


# --- Step 1: Create a weight ---
# This is the only "parameter" in our tiny network.
# It starts at a random value — the network knows nothing yet.
# requires_grad=True tells PyTorch to watch this tensor.
# When we call .backward() later, it will know how to compute
# the gradient for this weight automatically.

weight = torch.tensor([0.5], requires_grad=True)

print(f"Starting weight: {weight.item():.4f}")
# We want this to eventually become 3.0 — but it starts at 0.5.
# The network has no idea yet.


# --- Step 2: Create an input ---
# This is the data we are feeding in.
# It does NOT need requires_grad — we are not learning the input,
# we are learning the weight.

x = torch.tensor([4.0])

print(f"Input: {x.item():.4f}")


# --- Step 3: The forward pass ---
# The network's prediction is: weight × input
# This is the simplest possible "layer" — one multiplication.
# A real network does this thousands of times across many layers.

prediction = weight * x

print(f"Prediction: {prediction.item():.4f}")
# With weight=0.5 and input=4.0, prediction = 2.0
# But the correct answer is 3 × 4.0 = 12.0
# So the network is very wrong right now.


# --- Step 4: The correct answer ---
# This is what the network SHOULD have predicted.
# In a real dataset, this comes from your training data.
# Here we hardcode it: input × 3 is always the target.

target = x * 3

print(f"Target (correct answer): {target.item():.4f}")


# --- Step 5: Calculate the loss ---
# We use Mean Squared Error (MSE) loss.
# MSE = (prediction - target) squared
#
# Why squared?
#   - A negative error and a positive error are equally bad.
#     Squaring removes the sign — both become positive.
#   - Squaring also punishes large errors more than small ones.
#     An error of 10 becomes 100. An error of 1 stays 1.
#     This pushes the network harder when it is very wrong.

loss = (prediction - target) ** 2

print(f"Loss: {loss.item():.4f}")
# With prediction=2.0 and target=12.0:
# loss = (2.0 - 12.0)^2 = (-10.0)^2 = 100.0
# That is a large loss — which makes sense, the network is very wrong.


# --- What happens next? (preview) ---
# We now have a loss — a single number measuring how wrong the network was.
# Next session: we call loss.backward() to compute the gradient,
# then use that gradient to nudge the weight closer to 3.0.
# That nudge is called a training step.