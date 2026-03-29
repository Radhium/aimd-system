Let's start.
What is a forward pass?
You have a network. It has weights inside it — numbers it learned, or in the beginning, random numbers it was given.
A forward pass is simply: feed data in one end, let it travel through every layer, get a prediction out the other end.
That's it. Forward — because the data moves in one direction, from input to output. No learning happens yet. Just calculation.
Think of it like a kitchen that hasn't been trained yet. You put ingredients in, it produces a dish. The dish might be terrible. But it produced something. That something is the prediction.

What is a loss function?
Now you have a prediction. You also have the correct answer — the thing the network should have said.
A loss function takes those two things — prediction and correct answer — and returns a single number: how wrong was the prediction?

Loss of 0 — perfect prediction
Loss of 10 — very wrong
Loss of 0.003 — nearly right

You already have loss in your glossary from Session 2. Today you will see it calculated in actual code for the first time.

The simplest possible example
Before we touch anything language-model related, I want you to see the full loop in the smallest possible form.
Here is the situation we will model in code:

The network has one input and one output
It is trying to learn one simple fact: multiply the input by 3
It starts with a random weight — so its first prediction will be wrong
We calculate the loss — how wrong it was
That's where today stops. We are not training yet. Just: input → prediction → loss.

The training — using that loss to improve the weights — is the next session.

Here is the code. Create a new file in your project folder called forward_pass.py and copy this in.
pythonimport torch

# ============================================================

# FORWARD PASS AND LOSS — the simplest possible example

#

# Goal: the network should learn to multiply any input by 3.

# Today we do NOT train it. We just:

# 1. Make a prediction (forward pass)

# 2. Measure how wrong it was (loss)

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

prediction = weight \* x

print(f"Prediction: {prediction.item():.4f}")

# With weight=0.5 and input=4.0, prediction = 2.0

# But the correct answer is 3 × 4.0 = 12.0

# So the network is very wrong right now.

# --- Step 4: The correct answer ---

# This is what the network SHOULD have predicted.

# In a real dataset, this comes from your training data.

# Here we hardcode it: input × 3 is always the target.

target = x \* 3

print(f"Target (correct answer): {target.item():.4f}")

# --- Step 5: Calculate the loss ---

# We use Mean Squared Error (MSE) loss.

# MSE = (prediction - target) squared

#

# Why squared?

# - A negative error and a positive error are equally bad.

# Squaring removes the sign — both become positive.

# - Squaring also punishes large errors more than small ones.

# An error of 10 becomes 100. An error of 1 stays 1.

# This pushes the network harder when it is very wrong.

loss = (prediction - target) \*\* 2

print(f"Loss: {loss.item():.4f}")

# With prediction=2.0 and target=12.0:

# loss = (2.0 - 12.0)^2 = (-10.0)^2 = 100.0

# That is a large loss — which makes sense, the network is very wrong.

# --- What happens next? (preview) ---

# We now have a loss — a single number measuring how wrong the network was.

# Next session: we call loss.backward() to compute the gradient,

# then use that gradient to nudge the weight closer to 3.0.

# That nudge is called a training step.

Run that and tell me what the four printed numbers are. Before you run it — what do you expect the loss to be, given the prediction and target you can already see in the comments?I can already see it in the comments... which is 100, but the output it is showing is Loss: 100.0000 which is just more precise. but now my question is, what do you mean by this...# We want this to eventually become 3.0 — but it starts at 0.5. # The network has no idea yet.we put the data to be 0.512:21 pmBoth of those are sharp observations. Let me take them in order.

On point 1 — yes, exactly right. 100 and 100.0000 are the same number. PyTorch prints tensors with four decimal places by default. You read the comments and predicted correctly before running. That is the right habit.

On point 4 — this is the most important question you could ask right now.
You did not put the data to be 0.5. That distinction is everything.
There are two completely different kinds of numbers in a neural network and beginners conflate them constantly. Let me separate them clearly.

The input is the data coming from outside. In our example that is x = 4.0. You chose it. It represents something real — a measurement, a word, a pixel. The network does not own it. It just receives it.
The weight is a number the network owns internally. In our example that is weight = 0.5. Nobody chose 0.5 meaningfully — it was essentially a placeholder. In a real network, weights start randomly initialised. The network has never seen data before. It has no idea what the right number is. So it starts somewhere arbitrary.

Here is the analogy that might make it click.
Imagine you are learning to guess someone's age from their photo. On day one you have no experience. Someone shows you a photo and says "guess." You say 35. You are wrong — they are 60. Someone tells you that. Next photo, you adjust a little. Over thousands of photos you get better.
The photo is the input. It changes every time — a new person each round.
Your internal intuition — whatever mental model you are using to make the guess — that is the weight. It starts naive. It gets updated every round based on how wrong you were.
0.5 is the naive starting guess. The network does not know it should be 3.0 yet. That is the entire point of training — to discover that 3.0 is the right weight by being wrong repeatedly and correcting.

So to directly answer your question:
We did not choose 0.5 because 0.5 is meaningful. We chose it as a starting point that is clearly wrong, so you can watch the weight move toward 3.0 over training steps. Next session, you will see it move.
Does that separate the two clearly — input versus weight?okay, now I get it12:24 pmGood. Let's lock that distinction into the glossary before we move on — it is one of those things that looks obvious now but gets blurry later when there are thousands of weights and batches of inputs flying around.
Here is what I want you to do before next session. Just sit with this one question — no code, no writing needed:
If the weight started at 3.0 instead of 0.5 — what would the loss be?
Work it through with the numbers from the code. You have everything you need.
Tell me the answer next time we pick up, and then we move straight into the backward pass — where the network actually uses the loss to fix the weight.
If the weight started at 3.0 instead of 0.5 — what would the loss be? the loss would be 0.
On point 1 — exactly right. Weight 3.0 × input 4.0 = prediction 12.0. Target is also 12.0. (12.0 − 12.0)² = 0. Perfect prediction, zero loss. You worked that out cleanly.
Good. Let's keep moving.
You just established that loss = 0 when the weight is perfect. That is the destination. Now the question is: how does the network find its way there, starting from the wrong place?

The backward pass — what it actually is
You already know what .backward() does from Session 4. It reads the computational graph backwards and computes gradients.
But in Session 4 you saw it on a raw equation. Now you will see it in the context that actually matters — after a loss has been calculated.
The gradient on the weight answers one specific question:
If I increase the weight slightly, does the loss go up or down, and by how much?
That is all it is. A number that tells you which direction makes things worse — so you walk the other way.

The nudge
Once you have the gradient, you update the weight like this:
new_weight = old_weight - (gradient × learning_rate)
The learning rate is a small number — typically something like 0.01. It controls how big a step you take. Too large and you overshoot. Too small and training takes forever.
You subtract because you want to go downhill — opposite to the gradient.
This single update is called a training step. One forward pass, one loss calculation, one backward pass, one weight update. Repeat thousands of times — that is training.

Now the code. Open forward_pass.py and replace everything in it with this:
pythonimport torch

# ============================================================

# THE BACKWARD PASS — computing the gradient and updating the weight

#

# We extend the previous example. Same setup:

# - Network should learn to multiply input by 3

# - Weight starts at 0.5 (wrong)

# - Input is 4.0

#

# Today we add:

# - loss.backward() → computes the gradient

# - A manual weight update using that gradient

# - Print the loss before and after to confirm it dropped

# ============================================================

# --- Setup (same as before) ---

weight = torch.tensor([0.5], requires_grad=True)
x = torch.tensor([4.0])
target = x \* 3 # correct answer is always input × 3 = 12.0

print("=== Before training step ===")
print(f"Weight: {weight.item():.4f}")

# --- Forward pass ---

# Data moves through the network. Prediction is made.

prediction = weight \* x
loss = (prediction - target) \*\* 2

print(f"Prediction: {prediction.item():.4f}")
print(f"Target: {target.item():.4f}")
print(f"Loss: {loss.item():.4f}")

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
weight -= learning_rate \* weight.grad

print(f"\n=== After training step ===")
print(f"Weight: {weight.item():.4f}")

# --- Forward pass again with the updated weight ---

# Run the exact same prediction with the new weight.

# The loss should be lower than before.

prediction = weight \* x
loss_after = (prediction - target) \*\* 2

print(f"Prediction: {prediction.item():.4f}")
print(f"Loss: {loss_after.item():.4f}")
print(f"\nLoss went from {100.0:.4f} → {loss_after.item():.4f}")
print("One small step in the right direction.")

Before you run it — two questions:

After one training step, do you expect the weight to go up or down from 0.5? (Think about which direction gets it closer to 3.0.)
Do you expect the loss to go up or down?

Tell me your predictions, then run it.if the weight goes up, the loss gets down. if the weight goes down, the loss gets up.but my question is, is it possible to predict weather weight would go up or down ? would it not be random ? or something the network decides ?12:38 pmYour predictions on 1 and 2 are exactly right.
On point 3 — this is a great question, and the answer is: it is not random at all. The gradient tells you exactly which way to go.
Here is why.
The gradient is a number. That number has a sign — it is either positive or negative.

If the gradient is positive — increasing the weight would increase the loss. So you subtract. Weight goes down.
If the gradient is negative — increasing the weight would increase the loss in the other direction. So you subtract a negative, which means weight goes up.

You subtract the gradient either way. The direction takes care of itself automatically from the sign.
In our specific case — weight is 0.5, target weight is 3.0, so the weight needs to go up. That means the gradient will come out negative. You can verify that when you run the code — check the sign on the printed gradient.
The network does not "decide" anything. It is pure arithmetic. The gradient is calculated, the sign tells you the direction, the learning rate controls the size of the step. No randomness anywhere in this part.
The only randomness in a neural network is at the very beginning — the initial weights are set randomly. After that, every update is deterministic. Gradient in, weight update out.
Now run the code and tell me three things:

What was the gradient — and was its sign what you expected?
What did the weight move to?
What did the loss drop to?
the gradient was negative.the weight increased to 1.3loss droped to 46.240012:41 pmAll three are exactly right, and the signs all make sense together:
Gradient was negative → subtracting a negative → weight went up. The math took care of the direction automatically, exactly as predicted.
Weight moved from 0.5 to 1.3. Still far from 3.0 — but it moved the right way on the very first step, with no human guidance. Just arithmetic.
Loss dropped from 100 to 46.24. Nearly halved in one step. That is the backward pass working.

Now here is the thing worth sitting with for a moment.
You just did one training step. Manually. With one weight.
A real language model does this with hundreds of millions of weights, all at once, thousands of times, on batches of text. But the core operation is identical to what you just ran. Every single weight gets a gradient. Every single weight gets nudged. Loss drops a little. Repeat.
That is all training is. What you just did is not a simplification of the real thing — it is the real thing, just with one weight instead of millions.

You are now one concept away from having seen the complete training loop:

✅ Forward pass
✅ Loss calculation
✅ Backward pass
✅ Weight update

The one thing missing is the loop — repeating those four steps automatically until the weight reaches 3.0. That is the next and final piece of Phase 2 before we move to neural networks.
Do you want to do that now, or stop here for today?we continue12:42 pmGood. This is the satisfying part — you are about to watch the network actually learn.

The training loop
You have done one step manually. A training loop is just that same step, repeated, with one small addition — after each weight update, you have to clear the gradient.
Here is why. PyTorch accumulates gradients by default — meaning if you call .backward() twice without clearing, the second gradient gets added on top of the first. That would corrupt your update. So before each new forward pass, you zero out the gradient from the previous step.
The full loop looks like this every iteration:

Forward pass → get prediction
Calculate loss
.backward() → compute gradient
Update weight
Zero the gradient — ready for next iteration

That's it. Repeat until loss is small enough.

Replace everything in forward_pass.py with this:
pythonimport torch

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
target = x \* 3 # correct answer is always 12.0

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
    if step % 10 == 0:
        print(f"Step {step:3d} | Weight: {weight.item():.4f} | Loss: {loss.item():.4f}")

# --- Final result ---

print(f"\nFinal weight: {weight.item():.4f}")
print(f"Target weight: 3.0000")
print(f"Final loss: {(weight \* x - target).item() \*\* 2:.6f}")

Before you run it — one prediction:
After 100 steps, where do you think the weight will end up? Will it reach exactly 3.0, get close but not quite, or overshoot past it?
Tell me your prediction, then run it.at first I thought, we will get 3 but by first closing it, and then it might overshoot and then again come back to 3. but seeing all 10 steps, no, that did not happen, it was only closing until it got 3, but even more, it did not cross 3. next I changed the code to see all 100 steps.... these part was drastical...Step 24 | Weight: 2.9998 | Loss: 0.0000
Step 25 | Weight: 2.9999 | Loss: 0.0000
Step 26 | Weight: 2.9999 | Loss: 0.0000
Step 27 | Weight: 2.9999 | Loss: 0.0000
Step 28 | Weight: 3.0000 | Loss: 0.0000 from step 25 to 27, even though the loss were 0, it was being more precise then just 4 decimals, which it was not even showing12:50 pmYou did not just run the code — you investigated it. Changing the print interval to see all 100 steps was not in the instructions. That was your own curiosity. That is exactly the right instinct for this kind of work.
Now let me address each observation.

On point 1 — the overshoot idea
Your instinct was reasonable. Overshooting absolutely can happen — and it does, when the learning rate is too large. If you had set learning_rate = 0.9 instead of 0.01, the weight would have jumped past 3.0, overcorrected back, jumped again, and potentially never settled. That behaviour has a name: divergence. The learning rate being small (0.01) is what kept the steps smooth and one-directional here.

On point 2 and 3 — it only closed in, never crossed
Correct. With a small learning rate and a simple problem like this, the weight approaches the target from one side and asymptotically closes in. Each step the gradient gets smaller because the loss gets smaller, so the nudge gets smaller too. The steps shrink naturally as you get closer.

On points 4 and 5 — the most important observation
This is the sharpest thing you noticed today. Loss showed 0.0000 at step 25, but the weight was still 2.9999 — not yet 3.0000. The loss was not actually zero. It was something like 0.000016, which rounds to 0.0000 at four decimal places. The weight kept moving because the gradient was still non-zero, just tiny. It was still learning — just at a scale the display was not showing you.
By step 28 the weight hit 3.0000 — which is again four decimal places, meaning 3.00001 would also display that way. In practice the network gets close enough that the difference is meaningless. This is normal. Networks rarely reach the mathematically perfect answer — they get close enough that the remaining error does not matter.

You have now seen the complete training loop work end to end. Let me show you what you have built in the last three sessions laid out as a whole:
One complete training cycle, which you now understand every line of:

Forward pass — prediction is made
Loss — how wrong it was, measured as a number
Backward pass — gradient computed automatically
Weight update — nudge in the right direction
Zero gradient — clear for next step
Repeat

This is the engine inside every neural network that has ever been trained. GPT-4, image recognition, everything. Same loop, more weights, more data.
