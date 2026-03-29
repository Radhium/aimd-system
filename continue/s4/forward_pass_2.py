import torch

# ============================================================
# THE BACKWARD PASS — computing the gradient and updating the weight
#
# We extend the previous example. Same setup:
#   - Network should learn to multiply input by 3
#   - Weight starts at 0.5 (wrong)
#   - Input is 4.0
#
# Today we add:
#   - loss.backward()  → computes the gradient
#   - A manual weight update using that gradient
#   - Print the loss before and after to confirm it dropped
# ============================================================


# --- Setup (same as before) ---

weight = torch.tensor([0.5], requires_grad=True)
x = torch.tensor([4.0])
target = x * 3   # correct answer is always input × 3 = 12.0

print("=== Before training step ===")
print(f"Weight:     {weight.item():.4f}")


# --- Forward pass ---
# Data moves through the network. Prediction is made.

prediction = weight * x
loss = (prediction - target) ** 2

print(f"Prediction: {prediction.item():.4f}")
print(f"Target:     {target.item():.4f}")
print(f"Loss:       {loss.item():.4f}")


# --- Backward pass ---
# PyTorch reads the computational graph backwards.
# It computes: how much does the loss change if I nudge the weight?
# The answer is stored in weight.grad

loss.backward()

print(f"\nGradient on weight: {weight.grad.item():.4f}")
# This number tells you the slope at this point.
# Positive gradient → increasing the weight increases the loss.
# We want to DECREASE the loss, so we move in the OPPOSITE direction.


# --- Weight update ---
# This is the learning step. We nudge the weight slightly
# in the direction that reduces loss.
#
# learning_rate controls the step size.
# Too large → overshoot the target.
# Too small → takes forever to get there.
#
# torch.no_grad() tells PyTorch: do NOT record this operation.
# We are not computing a prediction here — just doing arithmetic
# on the weight itself. We don't want this in the computational graph.

learning_rate = 0.01

with torch.no_grad():
    weight -= learning_rate * weight.grad

print(f"\n=== After training step ===")
print(f"Weight:     {weight.item():.4f}")


# --- Forward pass again with the updated weight ---
# Run the exact same prediction with the new weight.
# The loss should be lower than before.

prediction = weight * x
loss_after = (prediction - target) ** 2

print(f"Prediction: {prediction.item():.4f}")
print(f"Loss:       {loss_after.item():.4f}")
print(f"\nLoss went from {100.0:.4f} → {loss_after.item():.4f}")
print("One small step in the right direction.")