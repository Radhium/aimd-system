# Phase: Phase 3 — First neural network (in progress)

_Last session (Session 6)_: You built and ran network.py — a two-layer network with 25 parameters that learned y = 3x. Loss fell from 12.4 to 0.004. You understood bias, ReLU, nn.Module, nn.Linear, Adam, and correctly read tensor shape [8, 1] as "8 neurons, 1 input each."

_Today's goal_: Multiple input features — what changes when a neuron receives many numbers instead of one. What a hidden layer is actually learning.
Let's begin.

---

_What changes when there are many inputs?_

Until now, every neuron in your network has received one number — a single x value. The rule it learned was simple: y = 3x.

But in real problems — and eventually in your language model — each input will be a vector: many numbers at once. A word embedding, for example, might be 128 numbers long.

So the question is: what does a neuron actually do with many inputs?

_The neuron's job, spelled out_

With one input, a neuron computes:

`z = (weight × input) + bias`

With many inputs, it computes the exact same kind of thing — just once per input, then adds them all up:
`z = (w₁ × x₁) + (w₂ × x₂) + (w₃ × x₃) + ... + bias`

Each input gets its `own weight`. The neuron multiplies each input by its weight, sums everything together, adds its bias, then passes the result through its activation function.

This is called a `weighted sum (or dot product)`.

_A concrete example before we go to code_

Imagine you're trying to predict house price. Your inputs are:

```
|x₁|x₂|x₃|
|size (m²)|number of rooms| distance to city (km)|
```

One neuron might learn:

```
w₁ = +0.8 → size matters a lot, positively
w₂ = +0.3 → more rooms helps, but less so
w₃ = −0.6 → further away → lower price
bias = +0.1 → a small baseline offset
```

The neuron reads all three numbers, weighs each one by its importance, sums them up, and produces a single number. That number goes to the activation function.

_The neuron has collapsed many numbers into one signal._ That's its whole job.

---

What does `nn.Linear` look like with multiple inputs?

In code, nothing changes except two numbers:

````
pythonnn.Linear(in_features=3, out_features=8)
```
This means: "each neuron receives 3 inputs. There are 8 neurons."

PyTorch automatically creates a **weight matrix** of shape `[8, 3]` — 8 neurons, each with 3 weights. Plus 8 biases, one per neuron.

Total parameters in just this one layer: `(8 × 3) + 8 = 32`.

You already read weight matrices last session. `[8, 1]` was 8 neurons with 1 weight each. `[8, 3]` is 8 neurons with 3 weights each. Same shape logic.

---

### What is a hidden layer actually learning?

This is the important question.

When you have:
```

Input → Hidden layer → Output layer
````

The `hidden layer` is not trying to produce the final answer. It is trying to produce `useful intermediate representations` — features that make the output layer's job easier.

Here is an analogy. Suppose you want to predict whether a day is good for a picnic. Your raw inputs are: temperature, wind speed, humidity, cloud cover.

The hidden layer might learn to combine those into signals like:

```
"this is a hot and dry day" (high temp, low humidity)
"this is a windy and overcast day" (high wind, high clouds)
```

The output layer then just reads those and decides: picnic or not?

The hidden layer learned to `group` and `transform` the raw inputs into something more meaningful. It invented its own intermediate features — ones that were never in the data explicitly.
_This is what makes depth powerful_. Each layer can build on the previous layer's abstractions.

In your language model, later on: the first attention layers might learn to group letters into word patterns. Deeper layers might learn to group words into grammatical roles. The final layers predict the next token. Nobody programmed those intermediate representations — the network learned them.

---

_Warm-up question before we write code_

Here is a single neuron with two inputs:

```
w₁ = 2.0, w₂ = −1.0, bias = 0.5
x₁ = 3.0, x₂ = 4.0
Activation: ReLU
```

Step 1: Compute z = (w₁ × x₁) + (w₂ × x₂) + bias
Step 2: Apply ReLU to z
What is the final output? Work through it — I'll wait.

```
the final value is 2.5
```

Correct. Let's check the working:
z = (2.0 × 3.0) + (−1.0 × 4.0) + 0.5
z = 6.0 − 4.0 + 0.5
z = 2.5
ReLU(2.5) = 2.5 — positive, so it passes through unchanged.
No hesitation, no errors. You're ready for the code.

---

_The network we're building today_

Last session: one input feature (x), learned y = 3x.

Today: two input features (x₁, x₂), learned rule: y = 2x₁ + 3x₂

The network will never be told that rule. It has to discover it from data. We will see whether the hidden layer's weights reflect the 2 and 3 coefficients after training.

Here is the architecture:

```
Input: 2 features
↓
Hidden layer: nn.Linear(2, 8) → ReLU ← 8 neurons, each sees both inputs
↓
Output layer: nn.Linear(8, 1) ← collapses 8 signals into 1 prediction
↓
Output: 1 number (predicted y)
```

# The code — network2.py

Create a new file called network2.py in C:\projects\myLLM\. Copy this in:

```
pythonimport torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── 1. Generate training data ───────────────────────────────────────────────

# The rule we want the network to discover: y = 2*x1 + 3*x2

# We never tell the network this formula — it must learn it from examples.

torch.manual_seed(42) # makes results reproducible

# 200 training examples, each with 2 input features

X = torch.rand(200, 2) # shape [200, 2] — 200 rows, 2 columns (x1 and x2)

# True targets: apply the hidden rule

# X[:, 0] means "all rows, column 0" → that's x1

# X[:, 1] means "all rows, column 1" → that's x2

y = 2 _ X[:, 0] + 3 _ X[:, 1] # shape [200]
y = y.unsqueeze(1) # reshape to [200, 1] so it matches the output shape

# ── 2. Define the network ───────────────────────────────────────────────────

class TwoInputNet(nn.Module):
def **init**(self):
super().**init**() # Hidden layer: takes 2 inputs, produces 8 signals
self.hidden = nn.Linear(in_features=2, out_features=8)
self.relu = nn.ReLU() # Output layer: takes 8 signals, collapses to 1 prediction
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

# Expected: (2*8 + 8) + (8*1 + 8) = 24 + 16 = 40

# ── 3. Loss and optimizer ───────────────────────────────────────────────────

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ── 4. Training loop ────────────────────────────────────────────────────────

losses = []

for epoch in range(1000): # Forward pass
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
```

Run it. Then report back:

1. What did it print for Total parameters?
2. What was the loss at Epoch 0 and at Epoch 900?
3. What did it predict for the test input x1=1.0, x2=1.0?
4. Paste the hidden layer weight printout — just the numbers, doesn't need to be tidy.

```
Total parameters: 33
Epoch 0 | Loss: 7.105975
Epoch 100 | Loss: 0.118081
Epoch 200 | Loss: 0.015510
Epoch 300 | Loss: 0.002395
Epoch 400 | Loss: 0.001462
Epoch 500 | Loss: 0.001157
Epoch 600 | Loss: 0.000906
Epoch 700 | Loss: 0.000700
Epoch 800 | Loss: 0.000535
Epoch 900 | Loss: 0.000406
── Hidden layer weights (shape [8, 2]) ──
tensor([[0.3611,  1.3119],
        [ 0.5609,  0.8648],
        [-0.1740, -0.6403],
        [-0.6649, -0.3380],
        [-0.4756,  0.0831],
        [-0.2058, -0.2766],
        [ 1.2190,  1.1315],
        [ 0.8389,  0.3663]])
── Output layer weights (shape [1, 8]) ──
tensor([[1.1152,  0.4559, -0.0616, -0.3275, -0.0018,  0.0900,  0.9336,  0.2484]])
Test: x1=1.0, x2=1.0 → predicted: 5.0039 | true: 5.0
```

Good catch on the parameter count — there's a small discrepancy to explain first.
The code comment said to expect 40 parameters, but you got 33.
Let's count properly:

```
Hidden layer: nn.Linear(2, 8) → 2×8 weights + 8 biases = 24
Output layer: nn.Linear(8, 1) → 8×1 weights + 1 bias = 9
Total: 33 ✓
```

The comment had a typo — it wrote 8 output biases instead of 1. Your number is right.

Now the interesting part.
_The network predicted 5.0039 for a true answer of 5.0_. It never saw the formula y = 2x₁ + 3x₂. It discovered it from 200 examples, to within 0.04% accuracy.

---

_What the hidden layer weights are actually telling you_

Look at the weight matrix again. Each row is one neuron. Each neuron has two weights — one for x₁, one for x₂.

```
[ 0.3611, 1.3119] ← neuron 1: leans heavily on x2
[ 0.5609, 0.8648] ← neuron 2: responds to both, x2 more
[-0.1740, -0.6403] ← neuron 3: negative — inhibited by both
[-0.6649, -0.3380] ← neuron 4: negative
[-0.4756, 0.0831] ← neuron 5: mostly ignores x2
[-0.2058, -0.2766] ← neuron 6: weak, both negative
[ 1.2190, 1.1315] ← neuron 7: strong response to both
[ 0.8389, 0.3663] ← neuron 8: leans on x1
```

No single neuron learned "multiply x₁ by 2 and x₂ by 3." That's not how it works.

Instead, `each neuron carved out a different slice of the input space.` Some respond strongly to x₂. Some to x₁. Some are negative and get cancelled out by the output layer. The output layer then combined all 8 signals — weighting each neuron's contribution — and together they reconstruct the 2:3 relationship.

Look at the output layer weights alongside the hidden layer:

1. Neuron 1: x₂-heavy weights [0.36, 1.31], output weight 1.12 — big contribution, x₂ emphasis
2. Neuron 7: strong both [1.22, 1.13], output weight 0.93 — another large contributor
3. Neurons 3, 4: negative hidden weights, but output weights also negative [−0.06, −0.33] — two negatives multiply to a positive contribution
4. Neuron 5: near-zero output weight −0.002 — this neuron was essentially ignored

The network found a _distributed solution_. The 2:3 ratio is encoded across all 8 neurons collectively, not in any one place.

This is the core idea behind representation learning — the network learned its own intermediate features. Nobody designed them. They emerged from training.

_One question to check understanding_

Look at neuron 5: hidden weights [−0.476, 0.083], output weight −0.002.
Why do you think that neuron's contribution ended up near zero? What might have happened during training?
Take a guess — there's no trick here, just reasoning from what you know.

```
wait for this I did not understand.

That's the right call — which part lost you? Was it:Q: Which part lost you?
A: The neuron 5 question at the end8
```

Got it. Let me back up —

I jumped ahead of myself by asking that question before making sure the setup was clear.
Let's establish one thing first.

---

_The output weight is a volume knob_

Every neuron in the hidden layer produces a signal. But the output layer doesn't trust all 8 neurons equally. It has one weight per neuron — think of it as a volume knob for that neuron's signal.

```
Output weight near +1.0 → "I trust this neuron a lot, include its signal strongly"
Output weight near 0.0 → "I don't care about this neuron, turn it off"
Output weight negative → "This neuron's signal gets flipped before I use it"
```

Now look at neuron 5's output weight: −0.002

That is almost exactly zero. The output layer has turned neuron 5's volume down to nearly nothing.
So here is the question I was really asking — rephrased more simply:

_If the output layer is multiplying neuron 5's signal by −0.002, how much does neuron 5 actually affect the final prediction?_

Just reason it through — if you multiply any number by 0.002, what happens to it?

```
it becomes very small, negligible
there is one more thing, explain me, what are those values in the hidden layers.
[-0.6649, -0.3380],
[-0.4756, 0.0831],
[-0.2058, -0.2766],
[-0.4756, 0.0831],
[-0.2058, -0.2766],this layers... what exactly are those values.and then thisTest: x1=1.0, x2=1.0 → predicted: 5.0039 | true: 5.0meaning we did not test for this equation ?  y = 2*x1 + 3*x2.
```

Good — point 1 is exactly right. Neuron 5 is essentially switched off. The network decided it wasn't useful and ignored it.
Now your other questions. Let me take them one at a time.

---

_What are those numbers in the hidden layer?_

Take one row: [-0.6649, -0.3380]
This is one neuron. It has two weights because it receives two inputs.

```
−0.6649 is its weight for x₁
−0.3380 is its weight for x₂
```

When the network feeds in a training example, say x₁=0.5 and x₂=0.8, this neuron computes:

```
z = (−0.6649 × 0.5) + (−0.3380 × 0.8) + bias
z = −0.332 + −0.270 + bias
z = a negative number

```

Then ReLU fires — negative in, so output is 0. This neuron goes completely silent for that input.
For a different input where x₁ and x₂ are very small, z might be less negative, or the bias might push it positive. The neuron would then contribute something.
So those values are just the learned weights — the numbers PyTorch adjusted over 1000 epochs until the overall prediction was accurate. They're not hand-designed. They're whatever the optimizer landed on.

---

_Your question about the test_

Yes — y = 2*x1 + 3*x2 is exactly the equation we tested.
x1=1.0, x2=1.0 → true answer = (2 × 1) + (3 × 1) = 5.0
The network predicted 5.0039. So it got it right to within a rounding error.

The network was never shown that formula. During training it only saw 200 pairs of numbers: here are x₁ and x₂, here is the correct y. After 1000 epochs it had learned the relationship well enough that when we gave it a brand new input it had never seen, it returned almost exactly the right answer.

That's the whole point of training — learn a rule from examples, then apply it to new inputs.

Does that clear both questions up? Once you confirm, I have one small thing to add about why some hidden weights go negative — then we're done with this concept.

```
there is something still confuses me, what exactly are we trying to find here? is it the x1 and x2 or the values x1 and x2 be multiplied with ?
```

Really good question — this is the most important distinction in all of machine learning.
We already know x₁ and x₂. Those are the inputs. We give them to the network.
What we're trying to find are the weights — the numbers that x₁ and x₂ should be multiplied by.

---

Here is the clearest way to see it:

The hidden rule in our experiment was y = 2*x1 + 3*x2.

We generated 200 examples from that rule and handed them to the network. So the network saw things like:

```
x₁|x₂|correct y|
0.3|0.7|2.7|
0.9|0.1|2.1|
0.5|0.5|2.5|
```

The network was never told "multiply by 2 and 3." It only saw the input-output pairs. Its job was to figure out: what numbers do I need to multiply x₁ and x₂ by, so that my predictions match the correct y?

The answer it was searching for was: 2 and 3.

Training is the process of finding those multipliers. The weights start random. The loss tells the network how wrong it is. Backprop adjusts the weights. Repeat 1000 times. Eventually the weights settle on values that make the predictions correct.

One sentence summary:

`Inputs are given to us. Weights are what we're solving for.`

Does that land cleanly?

```
wait me frame it this way...

1. 2 and 3, are they x1 and x2 ?
2. or x1 and x2 are the values that we see in the hidden layers ?
3. because here it is written Test: x1=1.0, x2=1.0 → predicted: 5.0039 | true: 5.0
4. also explain me this what these are
5. ── Output layer weights (shape [1, 8]) ──
   tensor([[1.1152,  0.4559, -0.0616, -0.3275, -0.0018,  0.0900,  0.9336,  0.2484]])
```

Good — let me clear up all the confusion at once with a clean picture.

There are three completely separate things here

1. _x₁ and x₂ — the inputs_

   These are the numbers we feed into the network. In the test line:
   `Test: x1=1.0, x2=1.0`
   x₁=1.0 and x₂=1.0 are just two measurements we handed to the network. Like saying "here are today's temperature and humidity — now predict whether it will rain." We own these. We provide them.

2. _The hidden rule — 2 and 3_

   This is the secret formula we embedded in the training data: y = `2*x1 + 3*x2`
   The 2 and 3 are not x₁ and x₂. They are the correct multipliers — the answer the network is trying to discover. We never told the network "use 2 and 3." It had to figure that out from the examples.

3. _The weight values in the hidden layer — like_ `[-0.6649, -0.3380]`

   These are the network's learned multipliers — its best attempt at solving the problem. They are not 2 and 3 exactly, because the network has 8 neurons sharing the work in a distributed way. But together, they encode the 2:3 relationship.

   These are not x₁ and x₂. x₁ and x₂ are inputs that change every example. The weights are fixed after training — they are what got adjusted over 1000 epochs.

---

_The output layer weights_

`tensor([[1.1152,  0.4559, -0.0616, -0.3275, -0.0018,  0.0900,  0.9336,  0.2484]])`

There are 8 numbers here — `one per hidden neuron`. This is the output layer's volume knob for each neuron.

The output layer receives 8 signals from the 8 hidden neurons. It multiplies each signal by its weight and sums them all up to produce the final prediction.

```
Neuron 1's signal × 1.1152
Neuron 2's signal × 0.4559
Neuron 3's signal × −0.0616
... and so on
```

---

_The final prediction is the sum of all 8 of those._

The full picture in one diagram

```
You provide:          Network owns:         Network owns:
x₁ = 1.0    →    hidden weights      →    output weights    →    prediction
x₂ = 1.0         (shape [8, 2])           (shape [1, 8])         5.0039
                  learned over             learned over
                  1000 epochs              1000 epochs
```

Inputs come from outside. Weights are what training finds.
Does that separate the three things cleanly now?

```
okay I get it now, but now the thing is I did the same test with 20000 epoch
Epoch 19500 | Loss: 0.000000
Epoch 19600 | Loss: 0.000000
Epoch 19700 | Loss: 0.000004
Epoch 19800 | Loss: 0.000000
Epoch 19900 | Loss: 0.000000
── Hidden layer weights (shape [8, 2]) ──
tensor([[0.4014,  1.3485],
     [ 0.5410,  0.8552],
     [-0.1740, -0.6403],
     [-0.6649, -0.3380],
     [-0.4756,  0.0831],
     [-0.2058, -0.2766],
     [ 1.2625,  1.1711],
     [ 0.7513,  0.2888]])
── Output layer weights (shape [1, 8]) ──
tensor([[1.1025,  0.4156, -0.0616, -0.3275, -0.0018,  0.0900,  0.9414,  0.1917]])
Test: x1=1.0, x2=1.0 → predicted: 4.9999 | true: 5.0
and the results are almost still the same, why ?
while testing I noticed the whole code put pressure on cpu and not gpu why ? I think we made the environment exactly for that reason
```

Two excellent observations. Let me take them one at a time.

_Why did 20000 epochs barely change anything?_

Because the network had already learned the rule. By epoch 1000 the loss was 0.000406 — already extremely small. By epoch 1000 the weights had found a good solution. Running 19000 more epochs on top of that is like continuing to sharpen a pencil that is already sharp. There is almost nothing left to improve.

This is called `convergence` — the point where the loss has bottomed out and further training produces no meaningful change. The network has found its answer.

The weights barely shifted:

```
Neuron |1000 epochs|20000 epochs
Row 1 |[0.3611, 1.3119]|[0.4014, 1.3485]
Row 7 |[1.2190, 1.1315]|[1.2625, 1.1711]
```

Tiny movements. The solution was essentially already there.
Good instinct to test this. You just discovered empirically why blindly throwing more epochs at a problem is not always the answer.

---

Why is this running on CPU and not GPU?

Because we never told PyTorch to use the GPU. In the code, X and the model were created without .to("cuda"). PyTorch always defaults to CPU unless you explicitly move things over.

You're right that we set up CUDA exactly for this reason. The network is too small to feel the difference — 33 parameters trains in milliseconds regardless. But when we get to the real Transformer with millions of parameters, GPU will matter enormously.

We will add GPU support properly in Phase 6 when we write the real training loop. For now the learning networks are intentionally tiny — CPU is fine.

Both questions showed sharp thinking. You noticed the weights barely changed and asked why. You noticed CPU was being used and remembered why we set up CUDA. That kind of observation is exactly the right habit.

Ready to move on — or is anything still unclear?

```
ready to move on
```

Great. One concept left in Phase 3 before we move to the Transformer.What you've learned so far in Phase 3

1. A neuron computes a weighted sum of its inputs, adds a bias, passes through an activation
2. Multiple inputs just means more weights per neuron — one per input
3. Hidden layers learn distributed intermediate representations
4. The network found y = 2x₁ + 3x₂ without being told the formula

---

What's left — and it's a short one
What happens when the output isn't a number — it's a category?

Right now your network predicts a continuous number — price, y value, etc. But your language model's final job is different: given the previous words, which word comes next? That's not a number — that's a choice from a list of thousands.This introduces two new things:

1. `Softmax` — turns a list of raw scores into probabilities that sum to 1
2. `Cross-entropy loss` — the correct loss function when the output is a category, not a number

These are the last two concepts before Phase 4. They appear in literally every language model ever built — your model will use them too.Next session we cover both, write a small classification network, and then Phase 3 is complete and we move to the Transformer.Let me update the documents now before you close out.
