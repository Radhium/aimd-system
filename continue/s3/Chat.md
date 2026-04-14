# PyTorch Tensors

Let's start with the obvious question.
You already know what a NumPy array is. So why does PyTorch need its own thing?

NumPy arrays are excellent for math on the CPU. But they have two problems for our project:

- They cannot run on the GPU. A NumPy array lives in regular RAM. Your RTX 3050 cannot touch it.
- They have no memory. NumPy has no idea what operations you performed to get a result. It just gives you the answer and forgets everything.

`PyTorch tensors solve both problems.`

A PyTorch tensor is essentially a NumPy array with two superpowers added:
SuperpowerWhat it meansGPU-awareYou can move a tensor to your GPU with one line. PyTorch then runs all the math there — massively faster.Operation memoryPyTorch can watch every operation you do to a tensor and write it down. This written record is called the `computational graph.` You will need this for backpropagation.

That second superpower has a name: `autograd`.

We will get to it after you feel comfortable with the tensors themselves.

---

**What a tensor looks like in PyTorch**

You already know shapes from NumPy. Tensors have the same idea — 1D, 2D, 3D, same rules. The difference is just where they live and what they can do.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(x) # tensor([1., 2., 3.])
print(x.shape) # torch.Size([3])
```

Looks almost identical to NumPy. That is intentional — the PyTorch team designed it that way.

Your first exercise
Create a new file in VS Code. Call it tensors.py. Make sure you are in the venv — use the play button to run it.
Type this out — do not copy-paste:

```python
import torch

# Create a 1D tensor

a = torch.tensor([1.0, 2.0, 3.0])

# Create a 2D tensor (a matrix)

b = torch.tensor([[1.0, 2.0],
                   [3.0, 4.0]])

# Print both, and their shapes

print(a)
print(a.shape)

print(b)
print(b.shape)

# Check where the tensor lives

print(a.device)
```

Before you run it — tell me: what do you predict a.shape, b.shape, and a.device will print?

     a.shape = (3,)
     b.shape = (2,2)
     this I can't predict. 
     it says cpubut wait why cpu? should it not be in gpu then ?

Your shape predictions are exactly right. And your question about the device is exactly the right question to ask — that instinct is good.

---

**Why does it say CPU?**

Because PyTorch does not assume. When you create a tensor, it starts on the CPU by default — sitting in regular RAM, just like a NumPy array.
You have to explicitly move it to the GPU. PyTorch will never do this without you asking.
Think of it like this: your desk is the CPU. Your GPU is a separate workbench across the room — faster for big jobs, but you have to physically carry the work over there. PyTorch does not carry anything for you automatically.
The move is one line:

```
pythona_gpu = a.to("cuda")
print(a_gpu.device) # cuda:0
```

cuda:0 means: GPU number zero. You only have one GPU, so it will always be zero.

Add this to your tensors.py file and run it

```python
# Move tensor to GPU
a_gpu = a.to("cuda")
print(a_gpu.device)

# The original is still on CPU

print(a.device)
```

Two questions before you run it:

- What do you predict a_gpu.device prints?
- What do you predict a.device prints — does moving a to the GPU change the original, or does .to() make a copy?

Have a think, then run it and tell me what happened.

     I though a_gpu would be in gpucpuuser@Shree-Ji-PC MINGW64 /c/projects/myLLM

```
$ c:/projects/myLLM/venv/Scripts/python.exe c:/projects/myLLM/tensors.py
tensor([1., 2., 3.])
torch.Size([3])
tensor([[1., 2.],
     [3., 4.]])
torch.Size([2, 2])
cpu
cuda:0
cpu
but then again why cpu for the 1st one?

```

Your predictions were both correct. And I can see exactly what confused you — let me clear it up.

---

**Why does the first one still say CPU?**

Look at the order of your print statements in the code:

```python
print(a.device) # you printed this BEFORE moving it — says cpu
a_gpu = a.to("cuda")
print(a_gpu.device) # says cuda:0
print(a.device) # still says cpu — the original never moved
```

The first cpu you saw was a before the move. That was always going to say CPU — you created a on the CPU and had not moved it yet.

The second cpu at the bottom confirms your prediction: `.to("cuda")` does not move the original. It makes a copy on the GPU and gives it a new name — a_gpu. The original a stays exactly where it was.
So the output makes complete sense:

- a.device before the move → cpu ✓
- a_gpu.device after the move → cuda:0 ✓
- a.device after the move → still cpu ✓

The key thing to remember
`.to("cuda")` is always a copy. The original stays where it was. If you want the original to move, you would write:

```python
a = a.to("cuda")
```

Overwrite the name. Same idea as any other variable reassignment in Python.

Does that fully resolve the confusion, or is any part of it still fuzzy? Once you are solid on this, we move to the most important part — autograd.

     I want to know why it says "cuda:0" instead of gpu?

Great question. Short answer: because you can have more than one GPU.

Why cuda:0 and not just "gpu"?

Imagine a powerful research computer with four GPUs installed. PyTorch needs a way to talk about each one individually. So it numbers them:

```
cuda:0 — first GPU
cuda:1 — second GPU
cuda:2 — third GPU
cuda:3 — fourth GPU
```

"gpu" would be ambiguous — which gpu? cuda:0 is precise.

Your PC has one GPU — the RTX 3050. So you will always see cuda:0 and never anything else. But PyTorch uses the same naming system for everyone, whether they have one GPU or ten.

cuda is the name of NVIDIA's system for running code on the GPU — it is already in your glossary from Session 1. The :0 is just the index number. First item, index zero — same as Python lists.

```python
my_list = ["only item"]
my_list[0] # index 0 — the first one
```

Same idea. cuda:0 means "the CUDA system, GPU number zero."

Clear? If yes, we move to autograd — which is where PyTorch starts to feel genuinely different from NumPy.
clear

---

**Autograd — PyTorch's memory for operations**

Here is the problem autograd solves.

You already understand backpropagation conceptually — walk backwards through the chain of operations, multiply the derivatives at each step. But to do that, you need to remember what operations you actually performed. NumPy forgets everything the moment it gives you an answer.
PyTorch can remember. But it does not do this for every tensor — that would be slow and wasteful. You have to opt in.

How you opt in

When you create a tensor, you add one flag:

`pythonx = torch.tensor([2.0], requires_grad=True)`

`requires_grad=True means`: watch this tensor. Remember every operation that touches it.

From that moment, PyTorch starts building a record of everything you do to x. That record is called the computational graph.

**What the computational graph is**

Think of it like a receipt.

You go to a shop and buy three things. The receipt does not just show you the total — it shows every item, every price, exactly how the total was calculated. If you want to dispute one item, you can trace back through the receipt step by step.
The computational graph is PyTorch's receipt for every mathematical operation. When it is time to do backpropagation, PyTorch reads the receipt backwards — exactly like the chain rule — and calculates the gradient at each step.

---

Let us see it happen
Add this to your tensors.py file and run it:

```python
# A tensor PyTorch will watch
x = torch.tensor([3.0], requires_grad=True)

# Do some math to it

y = x \* 2
z = y + 1

# Ask PyTorch to calculate gradients

# This walks backwards through the graph

z.backward()

# The gradient of z with respect to x

print(x.grad)
Before you run it — think about this:
z = (x \* 2) + 1
```

If x changes by a tiny amount, how much does z change? In other words — what do you expect x.grad to print?

Take a guess, even if you are not sure. Then run it.

     it should be linear graph, this is all I can say by seeing the equation.and I cant predict that.it says tensor([2.])
     was I supposed to get the shape of x ?

Your instinct about it being linear is exactly right — and that is precisely why the answer is 2. Let me show you why.

Why 2?
You already know what a derivative is — how much does the output change when the input changes a tiny bit?
Look at the equation:

```
z = (x _ 2) + 1
If x goes from 3 to 4 — increases by 1 — what happens to z?
before: z = (3 _ 2) + 1 = 7
after: z = (4 _ 2) + 1 = 9
```

z increased by 2. So for every 1 unit x changes, z changes by 2.
That is the derivative. That is the gradient. PyTorch calculated it automatically by reading its receipt backwards — and got 2.
The +1 contributed nothing to the gradient. Adding a constant never affects the rate of change. The slope of z is entirely determined by the \_ 2.

---

About your question — were you supposed to get the shape?

No. x.grad gives you the actual gradient value, not the shape. You got exactly what you were supposed to get.
tensor([2.]) means: a tensor containing the value 2. The gradient of z with respect to x is 2. PyTorch computed that automatically — no manual chain rule from you.

This is the big deal

What just happened is remarkable. You did not calculate anything. PyTorch watched the operations, built the graph, walked backwards through it, and handed you the gradient.

This is what makes training a neural network possible. A real network has millions of weights. PyTorch tracks all of them simultaneously and computes every gradient in one .backward() call.

Does the connection between the derivative and the number 2 make sense now? Once you confirm that, we have one small final exercise and then I will update the documents.

     okay, now I get it. I suppose if I change the equation from y = x _ 3, by calculation the tensor should give me 3. correct ?

Exactly correct. Try it — change the line to y = x \_ 3 and run it.

Tell me what you get.

     tensor([3.])

Perfect. You just verified it yourself — that is the best way to know something is true.

---

**What you now understand**

Changing x _ 2 to x _ 3 changed the gradient from 2 to 3

The constant you multiply by is the gradient for a linear equation
PyTorch calculated it automatically by walking backwards through the graph

You did not touch a single derivative by hand

This is autograd. This is what will power the entire training loop of your language model later.
