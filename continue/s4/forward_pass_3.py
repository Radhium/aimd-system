import torch

# ============================================================
# THE TRAINING LOOP — watching the network learn over many steps
#
# Same problem: learn to multiply input by 3.
# Same single weight, same single input.
#
# Now we repeat the full cycle 100 times and watch
# the weight climb toward 3.0 and the loss fall toward 0.
# ============================================================


# --- Setup ---

weight = torch.tensor([0.5], requires_grad=True)
x = torch.tensor([4.0])
target = x * 3   # correct answer is always 12.0

learning_rate = 0.01


# --- The training loop ---

for step in range(100):

    # Step 1: Forward pass — make a prediction
    prediction = weight * x

    # Step 2: Calculate loss — how wrong were we?
    loss = (prediction - target) ** 2

    # Step 3: Backward pass — compute the gradient
    loss.backward()

    # Step 4: Update the weight — nudge it in the right direction
    with torch.no_grad():
        weight -= learning_rate * weight.grad

    # Step 5: Zero the gradient — clear it before the next iteration
    # If we don't do this, gradients from previous steps accumulate
    # and the updates become wrong.
    weight.grad.zero_()

    # Print progress every 10 steps so we can watch learning happen
    if step % 1 == 0:
        print(f"Step {step:3d} | Weight: {weight.item():.4f} | Loss: {loss.item():.4f}")


# --- Final result ---
print(f"\nFinal weight: {weight.item():.4f}")
print(f"Target weight: 3.0000")
print(f"Final loss:   {(weight * x - target).item() ** 2:.6f}")