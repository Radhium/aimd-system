# Phase 3 — Session 6 — What a neuron does

LEARNING session. No code yet. Concepts first.

What you already know
From Phase 2, you know:

A weight is a number the network adjusts during training
A forward pass multiplies weight × input to make a prediction
Loss measures how wrong that prediction was
Backward pass computes gradients, and we update the weight to reduce loss

In Phase 2, your entire "network" was one weight times one input. That's actually the skeleton of a neuron. Today we flesh it out.

Step 1 — What a single neuron actually is
In Phase 2 your forward pass looked like this:

`prediction = weight × input`

A real neuron does three things, not one:

```
step 1: z = (weight × input) + bias
step 2: output = activation_function(z)
```

Two new things appeared. Let's take them one at a time.

The bias
You have a weight. The weight controls how much the input matters. But what if the right answer is never zero, even when the input is zero?
Example: you're trying to predict temperature in Celsius from temperature in Fahrenheit. The formula is:
`C = (F − 32) × 5/9`
Even when F = 0, C is not 0. There's an offset baked in. That offset is the bias.
The bias is a number the neuron adds to its result before passing it on. It shifts the output up or down regardless of the input. Like the weight, it starts at an arbitrary value and gets adjusted during training.
So now your neuron looks like:
`z = (weight × input) + bias`
Both weight and bias are learnable. Both get gradients. Both get updated.

The activation function
Here is the problem without an activation function.
Say you stack two neurons:

```
neuron 1: z₁ = w₁ × input + b₁
neuron 2: z₂ = w₂ × z₁ + b₂
```

Substitute the first into the second:

```
z₂ = w₂ × (w₁ × input + b₁) + b₂
= (w₂ × w₁) × input + (w₂ × b₁ + b₂)
```

That is still just `some_weight × input + some_bias`. You stacked two neurons and got the same thing as one neuron. You can stack a hundred — you still get one straight line. A network of pure multiplication and addition, no matter how deep, has the same power as a single multiplication and addition.
The activation function breaks this. It is a simple mathematical function applied to z that introduces a bend — a non-linearity. After the activation, the result is no longer a straight line. Now stacking layers actually adds power.
The most common activation function today is called ReLU. It is almost insultingly simple:
`ReLU(z) = max(0, z)`
If z is positive, pass it through unchanged. If z is negative, output zero. That's the whole thing. One bend in the output. But that bend, repeated across millions of neurons and dozens of layers, is what lets a network learn complicated shapes — faces, language, logic.
So the full neuron is:

```
z = (weight × input) + bias
output = ReLU(z)
```

Visualising one neuron
Think of it like a pipe with a valve:

1. `Input` flows in
2. `Weight` controls how strongly the signal is amplified
3. `Bias` shifts the whole signal up or down
4. `Activation` decides whether to let the signal through at all (ReLU blocks negatives)

A single neuron is weak. But in a real network, one layer has many neurons running in parallel — each with its own weights and bias — each looking at the same input but picking up different patterns. Their outputs then feed into the next layer.

Warm-up question before we move to `nn.Module`
Don't run any code. Just think through this.
Imagine a neuron with:

```
weight = 2.0
bias = -5.0
activation function: ReLU
```

Two inputs arrive, one at a time:

```
Input A = 1.0
Input B = 3.0
```

What is the neuron's output for each input? Work through the arithmetic — z first, then apply ReLU.
Take your time. When you have answers, share them and we'll move on to nn.Module.

1. for input A, the neuron's output is -3. after ReLU  it's 0.
2. for input B, the neuron's output is 1. after ReLU it's 1.

Both correct. Working shown perfectly — z first, then ReLU applied. Input A got blocked (negative → zero), Input B passed through unchanged. That's exactly what ReLU does: it's a gate.

Step 2 — Why we need nn.Module
In Phase 2, you managed everything by hand:

```
pythonweight = torch.tensor([0.5], requires_grad=True)

# manually compute forward pass

# manually call .backward()

# manually update weight

# manually zero the gradient
```

That works for one weight. But a real network has thousands of weights and biases — one per neuron, per layer. Managing them all by hand would be chaos.
nn.Module is PyTorch's solution. It is a base class — a blueprint you inherit from — that handles the bookkeeping for you.
When you build a network by inheriting from nn.Module:

1. PyTorch automatically finds all your weights and biases — you don't have to list them
2. You can call model.parameters() to get every learnable number in one place
3. The optimizer (which we'll introduce today) can take model.parameters() and update everything in one line
4. Saving and loading the model later becomes trivial

You define two things inside an nn.Module:

1. **init** — where you declare the layers (you already know what **init** does from Phase 1)
2. forward — where you describe how data flows through those layers`
   PyTorch calls your forward method automatically when you pass data into the model.

Step 3 — nn.Linear

Before writing the network, one more piece: nn.Linear.
This is PyTorch's built-in neuron layer. When you write:

````
pythonnn.Linear(in_features=1, out_features=4)

`You are creating a layer with 4 neurons, each receiving 1 input. PyTorch automatically creates all the weights and biases for those 4 neurons — you don't declare them yourself. They are initialised to small random values, and they are automatically registered so `model.parameters()` finds them.`

`in_features` = how many numbers are coming in
`out_features` = how many neurons are in this layer (= how many numbers come out)

---

### The network we are about to build

Two layers:
```
input (1 number)
↓
Linear layer — 1 input, 8 neurons — ReLU
↓
Linear layer — 8 inputs, 1 output — no activation (raw prediction)
↓
output (1 number)
```
````

This is a proper neural network. Small, but real. It will learn the same task as Phase 2 — map one input to one target — but now using the full PyTorch idiom you'll use for everything going forward, including the Transformer.
Ready for the code? Say yes and I'll write network.py with full comments explaining every line.
`yes`
Here is network.py. Create this file in C:\projects\myLLM\ and copy it in exactly as written.

```
# network.py

# Phase 3 — First real neural network using nn.Module

# Task: learn to map input x to target y = 3x

# Same task as Phase 2, but now using PyTorch's proper building blocks

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── 1. Define the network ──────────────────────────────────────────────────────

class SimpleNetwork(nn.Module): # nn.Module is the base class for all PyTorch models. # Inheriting from it means PyTorch automatically tracks all weights and biases # declared inside this class — we don't manage them by hand.

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
print(f" {name}: shape {param.shape}")

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

# Does exactly what your manual (prediction - target)\*\*2 did in Phase 2,

# but handles batches cleanly and is ready for more complex use later.

# ── 3. Training data ───────────────────────────────────────────────────────────

# Same task as Phase 2: learn that output = 3 × input.

# We generate 100 input/target pairs spread across a range.

x_train = torch.linspace(-2, 2, 100).unsqueeze(1)

# linspace: 100 evenly spaced numbers from -2 to 2.

# unsqueeze(1): reshapes from shape (100,) to shape (100, 1).

# nn.Linear expects each input to be a row — shape (batch_size, features).

y_train = 3 \* x_train

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
x = torch.tensor([[val]]) # Shape must be (1, 1) — one example, one feature.

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
```

What to do

Create network.py in C:\projects\myLLM\
Activate the venv, open the file in VS Code, run it with the play button
Report back two things:

1. What the parameter list printed at the top looks like
2. What the five test predictions look like at the end

Don't interpret anything yet — just paste what you see and we'll read it together.

```user@Shree-Ji-PC MINGW64 ~/Desktop/aimd system (main)
$ c:/python314/python.exe "c:/Users/user/Desktop/aimd system/continue/s5/network.py"
Parameters inside the model:
layer1.weight: shape torch.Size([8, 1])
layer1.bias: shape torch.Size([8])
layer2.weight: shape torch.Size([1, 8])
layer2.bias: shape torch.Size([1])
Epoch 0 | Loss: 12.412384
Epoch 50 | Loss: 3.185372
Epoch 100 | Loss: 0.151693
Epoch 150 | Loss: 0.023579
Epoch 200 | Loss: 0.013065
Epoch 250 | Loss: 0.008729
Epoch 300 | Loss: 0.006277
Epoch 350 | Loss: 0.005011
Epoch 400 | Loss: 0.004286
Epoch 450 | Loss: 0.003763
Testing trained network:
input: -2.0 | predicted: -5.9228 | target: -6.0000
input: -1.0 | predicted: -3.0719 | target: -3.0000
input: 0.0 | predicted: -0.0502 | target: 0.0000
input: 1.0 | predicted: 3.0417 | target: 3.0000
input: 2.0 | predicted: 5.8583 | target: 6.0000
user@Shree-Ji-PC MINGW64 ~/Desktop/aimd system (main)
```

Two things to flag before we read the results.

First — you ran this with c:/python314/python.exe, not the venv. That's the wrong Python. It worked this time because network.py only uses standard libraries that happened to be installed, but this will cause silent problems later. Always activate the venv first and use the VS Code play button. That rule is in the permanent decisions for exactly this reason.

Second — the file is sitting on your Desktop, not in C:\projects\myLLM\. That's fine for now, but when we start creating multiple files that import each other, location will matter. We'll move everything to the right place at the start of the next Build session.

Now let's read the results
The parameter list:

```
layer1.weight: shape torch.Size([8, 1])
layer1.bias: shape torch.Size([8])
layer2.weight: shape torch.Size([1, 8])
layer2.bias: shape torch.Size([1])
```

This is nn.Module doing its bookkeeping. You declared two layers — it found all four groups of learnable numbers automatically. Notice the shapes:

1. layer1.weight is [8, 1] — 8 neurons, each with 1 weight
2. layer1.bias is [8] — one bias per neuron
3. layer2.weight is [1, 8] — 1 output neuron, receiving from 8
4. layer2.bias is [1] — one bias on the output neuron

Total learnable numbers: 8 + 8 + 8 + 1 = 25 parameters. Your Phase 2 network had 1. This one has 25. Still tiny — but it's a real network.

The loss curve:

```
Epoch 0 | Loss: 12.41
Epoch 50 | Loss: 3.19
Epoch 100 | Loss: 0.15
Epoch 150 | Loss: 0.02
Epoch 200 | Loss: 0.013
...
Epoch 450 | Loss: 0.0038
```

Loss fell from 12 to 0.004 — dropped by a factor of 3000. The big drop happened in the first 150 epochs, then it slowed. That shape is normal and healthy. The network learned fast at first, then made smaller and smaller corrections as it got close.

It didn't reach zero the way Phase 2 did. That's expected — Phase 2 had one weight learning one perfectly linear rule. This network has 25 parameters learning the same rule, with ReLU bends in the middle. It approximates the line very well but not perfectly.

**The test predictions:**

```
input: -2.0 | predicted: -5.923 | target: -6.000 (error: 0.077)
input: -1.0 | predicted: -3.072 | target: -3.000 (error: 0.072)
input: 0.0 | predicted: -0.050 | target: 0.000 (error: 0.050)
input: 1.0 | predicted: 3.042 | target: 3.000 (error: 0.042)
input: 2.0 | predicted: 5.858 | target: 6.000 (error: 0.142)
```

All five predictions are close. The sign is correct in every case. The network correctly learned that negative input → negative output, zero → near zero, positive → positive. That is the relationship `y = 3x` — it learned the direction and roughly the scale.

The small errors that remain are because 500 epochs wasn't quite enough to fully converge, and because ReLU networks approximate smooth functions rather than computing them exactly.

One question before we close
Look at the parameter list again:
`layer1.weight: shape torch.Size([8, 1])`

Why do you think the weight shape is [8, 1] and not [1, 8]? What does that arrangement tell you about how the layer is organised?

Think about what the numbers mean — 8 neurons, 1 input each — and how that maps to rows and columns. No code needed, just reasoning.

well, the weight is not shaped (1,8) as if that then, it would mean 1 neuron for 8 inputs. which is not the case, we have 1 input for 8 neurons dont worry about the environment and the files, thats intentional

1 — Exactly right. The shape encodes the structure directly — rows are neurons, columns are inputs. [8, 1] means 8 neurons each looking at 1 input. Flip it to [1, 8] and you'd have the opposite: 1 neuron receiving 8 inputs. You read a tensor shape and understood what it meant architecturally. That's the right instinct — you'll use this constantly when we get to the Transformer.

2 — Noted, won't flag it again.
