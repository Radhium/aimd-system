import torch
import matplotlib.pyplot as plt

# ============================================================
# THE TRAINING LOOP — watching the network learn over many steps
#
# Same problem: learn to multiply input by 3.
# Same single weight, same single input.
#
# Now we repeat the full cycle 100 steps and watch
# the weight climb toward 3.0 and the loss fall toward 0.
# ============================================================


# --- Setup ---

weight = torch.tensor([0.5], requires_grad=True)
x = torch.tensor([4.0])
target = x * 3   # correct answer is always 12.0

learning_rate = 0.01

# These lists record history so we can plot them after training
loss_history = []
weight_history = []


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
    weight.grad.zero_()

    # Record this step's values for the plot
    loss_history.append(loss.item())
    weight_history.append(weight.item())

    # Print progress every 10 steps
    if step % 10 == 0:
        print(f"Step {step:3d} | Weight: {weight.item():.4f} | Loss: {loss.item():.4f}")


# --- Plot ---
# Everything below is just drawing. The training is already done above.

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle("Training loop — one weight learning to multiply by 3", fontsize=13)

# Top chart: loss over time
ax1.plot(loss_history, color="#378ADD")
ax1.set_ylabel("Loss")
ax1.set_xlabel("Step")
ax1.set_title("Loss should fall toward 0", fontsize=11)

# Bottom chart: weight over time, with target line
ax2.plot(weight_history, color="#BA7517", label="Weight")
ax2.axhline(y=3.0, color="#999999", linestyle="--", linewidth=1, label="Target (3.0)")
ax2.set_ylabel("Weight")
ax2.set_xlabel("Step")
ax2.set_title("Weight should climb toward 3.0", fontsize=11)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()