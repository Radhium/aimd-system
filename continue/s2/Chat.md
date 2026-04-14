**Session 3 — LEARNING — The Chain Rule**

Good. Let's go.

What we're doing today and why it matters
The chain rule is the mathematical engine behind how neural networks learn. Every time a network trains, it uses the chain rule — thousands of times, automatically. You don't need to calculate it by hand in the project, but if you don't understand it, the training loop will feel like magic you can't control. Once you understand it, it clicks into place permanently.

---

**Start with what you already know**

From last session: a derivative tells you how much changing one number affects another number.

For example: if you turn up the oven temperature by 1 degree, the cake bakes 2 minutes faster. The derivative of bake-time with respect to temperature is −2.

Simple. One input, one output, one derivative.

---

**The problem the chain rule solves**

Now imagine this:

You turn up the oven temperature → that changes how fast the outside browns → which changes when you take the cake out → which changes how moist the inside is.

There's a chain of effects. You changed one thing at the start. Something changed at the end. But the connection runs through several steps in the middle.

The chain rule answers: if everything is connected in a chain, how does changing the very first thing affect the very last thing?

The simplest possible example

Let's say you have two functions chained together:

x → [double it] → y → [square it] → z

You put in x

It gets doubled to make y — so y = 2x

Then y gets squared to make z — so z = y²

Question: if you change x by a tiny amount, how much does z change?

You could answer this two ways:

- Way 1 — substitute and solve directly:
  Since y = 2x, then z = (2x)² = 4x². The derivative of 4x² is 8x. Done.
- Way 2 — the chain rule:
  1. How much does y change when x changes? The derivative of 2x is 2
  2. How much does z change when y changes? The derivative of y² is 2y

Chain rule says: multiply them together → 2 × 2y = 4y

Substitute back y = 2x → 4(2x) = 8x

Same answer. Both ways work.

---

**So why bother with the chain rule at all?**

Because Way 1 only works when the chain is short and simple enough to substitute.

In a neural network, you have dozens of layers, each doing their own operation. You cannot substitute them all into one giant formula — it would be impossibly complex.

The chain rule lets you go one step at a time, multiplying as you go. Each layer only needs to know two things:

- How sensitive is my output to my input?
- How sensitive is the final loss to my output? (passed back from the next layer)

Multiply those two together. Pass the result backwards. Repeat. That's backpropagation — but we'll get there. For now, just hold this:

`Chain rule = multiply the derivatives at each step as you work backwards`

The rule written simply

```
If z depends on y, and y depends on x:
dz/dx = (dz/dy) × (dy/dx)
```

Read it like this:

    "How does z respond to x?" equals
    "How does z respond to y?" times "How does y respond to x?"

The dy in the middle cancels out, like fractions. That's not just notation — it's the actual logic. The middle step is the bridge.

A concrete numbers example

```
Let x = 3.
y = 2x = 6
z = y² = 36
Now nudge x up slightly — say x = 3.001:
y = 2 × 3.001 = 6.002
z = 6.002² = 36.024004
z went from 36 to 36.024 — it changed by about 0.024 when x changed by 0.001.
That ratio is 0.024 / 0.001 = 24. But wait — we said the derivative was 8x = 8 × 3 = 24. Exact match.
```

The chain rule gave you the right number without needing to nudge anything manually.

Why this matters for neural networks — one sentence

In a network, the "chain" runs from the weights → through every layer → to the loss. The chain rule tells each weight: "if you change slightly, here is exactly how much the loss changes." That's the gradient. That's how learning happens.

Before I move to Python functions and loops — I want to check this landed.

In your own words: what is the chain rule doing, and why can't we just skip it and calculate the effect directly?

Take your time. There's no wrong answer here.

    chain rule I think, it just like doing the same thing as derivative but step by step, for safety cause for large parameters or values.again for large parameters it can become a headache to solve

That's exactly right — and you got there on your own reasoning, which matters more than memorising a definition.

Let me just sharpen the wording slightly so it's precise enough to build on:

What you said, tightened up:

The chain rule breaks a complex chain of effects into small steps, each giving you one derivative. Then you multiply them together. It's not a different thing from a derivative — it's just how you apply derivatives when things are connected in sequence.

For anything beyond a trivially short chain, substituting everything into one formula becomes unmanageable. The chain rule keeps each step local and simple — each part only needs to know about its immediate neighbours.

That's the right mental model. Hold onto it.

One small thing to add to your picture, because it'll matter later:
The chain rule doesn't just help with large values — it helps with depth. Even if the numbers are small, a network with 24 layers has 24 things chained together. You cannot write that as one formula. But you can walk back through it one step at a time, multiplying as you go. That's exactly what PyTorch does automatically every time you call .backward().

You'll see that in a few sessions and it'll make immediate sense because of what you just understood.

Ready to move to Python functions, loops, and classes — or do you want one more example of the chain rule with actual numbers before we leave it?
Ready to move to Python functions, loops, and classes.

---

**Python functions, loops, and classes**

Good. Let's go.

I want to know where you're starting from before I pitch this at the wrong level.

Have you written any Python before — even small scripts, a bit of copy-pasting, anything? Or is this genuinely from zero?
just basic syntax but can read a little, dont worry i'll you if I dont get anything.

Perfect. That's exactly the right starting point for this.

**Functions**

You probably know a function exists. Here's how to think about it properly.

A function is a named machine. You build it once, give it a name, and then you can run it whenever you want by calling that name.

```python
def double(x):
return x \* 2
```

- def — "I'm defining a machine"
- double — that's its name
- x — the input it expects (called a parameter)
- return — what it hands back when it's done

Call it like this:

```python
result = double(5) # result is now 10
```

That's it. A function takes something in, does something, hands something back.

Why this matters for the project: every layer of your neural network will be a function. Input goes in, output comes out. You'll be writing and reading these constantly.

**Loops**

A loop says: do this thing repeatedly.
The one you'll use most is for:

```python
for i in range(5):
print(i)

```

Output:

```

0
1
2
3
4
```

- range(5) — gives you the numbers 0, 1, 2, 3, 4
- i — the variable that takes each value in turn

Everything indented underneath runs once per value

You can loop over a list too:

```python
losses = [2.4, 1.9, 1.3, 0.8]

for loss in losses:
print(loss)
```

Each time around the loop, loss is the next item in the list.
Why this matters for the project: your training loop will literally be a for loop — one pass through the data per iteration, thousands of times.

**Classes**

This is the one that trips people up. Here's the honest version.
A class is a blueprint for building objects. An object is just a thing that holds data and has functions attached to it.
Think of it this way: a function is a machine. A class is a factory that builds machines — and each machine it builds remembers its own state.
Here's the simplest possible class:

```python
class Counter:
def __init__(self):
self.count = 0 # this is the object's memory

    def increment(self):
        self.count += 1

    def value(self):
        return self.count
```

- `__init__` — runs automatically when you create the object. Sets it up.
- self — refers to this specific object. Its own private data.
- self.count — a variable that belongs to this object and persists

Use it like this:

```python
c = Counter() # build one Counter object
c.increment() # count is now 1
c.increment() # count is now 2
print(c.value()) # prints 2
```

You can build two separate counters and they don't interfere with each other:

```python
a = Counter()
b = Counter()
a.increment()
a.increment()
b.increment()

print(a.value()) # 2
print(b.value()) # 1
```

Each object has its own self.count. They're independent.
Why this matters for the project: in PyTorch, every layer of your neural network — the attention layer, the feed-forward layer, the whole model — is a class. You'll define a class, and PyTorch creates objects from it during training. Understanding self and `__init__` is what makes that readable.

How these three things connect
In your project, you'll constantly see this pattern:

```python
class SomeLayer:
def __init__(self, size): # set up the layer's weights
self.weights = ...

    def forward(self, x):           # run input through the layer
        return x * self.weights     # (simplified)
```

The class defines the layer

`__init__` sets up its weights (its memory)
forward is the function that processes input

You'll call it inside a loop that runs once per training step

You just learned the three building blocks that make that readable.

Quick check — nothing complicated:

self is the thing that trips most people up. In your own words, what is self doing in a class? What would break if it wasn't there?

    I think the self is the parameter, for this specific function, or perhaps for this whole one class.and if my guess is correct, then I dont think anything would actually break, even if it was not there, but ig it will confuse the system, for multiple classes, because now it has to guess for which class what function I'm asking it to do

You're close — and your instinct on the second part is actually pointing at the right idea. Let me sharpen both.

**On what self is:**

It's not quite a parameter in the normal sense — it's a reference. self is how the object points back to itself. When you write self.count, you're saying "the count that belongs to this specific object", not any other one.

Think of it like a name tag. Every object wears its own name tag. self is how it reads its own tag.

**On what breaks without it:**

You were closer than you think, but the real answer is sharper. Without self, the data has nowhere to live between calls.
Watch what happens:

```python
def increment(self):
self.count += 1
```

vs if you tried to do it without:

```python
def increment():
count += 1 # which count? where is it? it doesn't exist here
```

The function runs and then forgets everything. There's no count attached to anything. It evaporates the moment the function ends.

self is what gives the object memory between calls. That's the real job it does.

Your intuition about multiple objects was correct — just one layer deeper than you stated. It's not that the system gets confused about which class — it's that without self, each object has no private storage at all. They can't hold anything. They'd all be stateless, empty shells.

That's functions, loops, and classes done. You're ready to write actual NumPy code next — which is where this stops being abstract and starts feeling real.

---

Shall we do that now, or is this a good place to stop for today?
start now NumPy in actual code

Good. This is where it gets real.

I'm going to give you three small exercises to run on your machine. Not copy-paste and move on — actually read each one, predict what you think will happen, then run it and tell me what you got.

First — activate your environment

Open a terminal and run:

```
C:\projects\myLLM\venv\Scripts\activate
```

Then start Python:

python

You should see >>> — that's Python waiting for input. You're now in an interactive session. Type directly into it.

**Exercise 1 — Arrays and shape**

Type this in, line by line:

```python
import numpy as np

v = np.array([1, 2, 3])
print(v)
print(v.shape)
```

Before you run it — what do you expect v.shape to print?
Tell me your prediction, then run it and tell me what actually appeared.

    what i expected was (3 x 1) but it's showing (3, )
    and one thing tell me how to write code and all in vs code, not in terminal.

Good observation — and that result is actually telling you something important. Let me explain it, then we'll move to VS Code.

**Why (3,) and not (3, 1)**

`(3,)` means one dimension, three elements. A pure 1D vector — just a line of numbers with no rows or columns concept yet.

`(3, 1)` would mean a 2D array — three rows, one column. That's a different shape, even though it holds the same numbers.

NumPy is being precise. A 1D list is not the same as a column. You'd have to explicitly tell NumPy you want (3, 1). We'll see that later when it matters.

Your instinct was reasonable — you were thinking visually. NumPy thinks structurally.

---

Moving to VS Code

This is the right call. The terminal is fine for one-liners but not for real work.

Set it up once:

```
Open VS Code
File → Open Folder → select C:\projects\myLLM
Create a new file in that folder — call it scratch.py
Now tell VS Code which Python to use — bottom right corner, click where it shows a Python version, and select the one from your venv. It will look something like:
C:\projects\myLLM\venv\Scripts\python.exe
Open the terminal inside VS Code: Terminal → New Terminal. You should see the venv already activated — the terminal prompt should start with (venv)
```

To run your file: press Ctrl+Shift+P, type Run Python File, hit enter. Or just press the play button ▷ top right.

Once you've got scratch.py open and the venv selected, put this in the file and run it:

```python
import numpy as np

v = np.array([1, 2, 3])
print(v)
print(v.shape)

m = np.array([[1, 2, 3],
              [4, 5, 6]])
print(m)
print(m.shape)
```

Before running — predict what m.shape will print. Tell me your guess, then tell me what you got.

    now it is a matrixit goes with like this right (rows, columns)?then, it should be (2, 3)but wait, then 1st one does not make sense... (3, )... what exactly that 3 is then ?

That's a really sharp question. You caught a genuine inconsistency and pushed on it instead of just accepting the answer. That's exactly the right instinct.

Here's the clean explanation:

- For a 2D array — (2, 3) means rows, columns. You got that exactly right.
- For a 1D array — there are no rows or columns yet. It's just a sequence of numbers. So NumPy reports only one number — how many elements are in that sequence. The 3 in (3,) means "3 elements". The comma after it is just NumPy's way of saying "this is a 1D shape, not a plain number".
  Think of it this way:

```
Array           Shape           What the numbers mean
[1, 2, 3]       (3,)            3 elements in a line
[[1,2,3],[4,5,6]](2, 3)         2 rows, 3 columns
A 3D array      (4, 2, 3)       4 grids, each 2 rows, 3 columns
```

Each new dimension adds a number to the left.

A 1D array hasn't become a grid yet — it's just a list. No rows, no columns. Just length.

Now run it and tell me what you got for m.shape. I'm expecting you already know the answer — I just want to confirm it printed what you predicted.

```user@Shree-Ji-PC MINGW64 /c/projects/myLLM
$ c:/python314/python.exe c:/projects/myLLM/scratch.py
[1 2 3]
(3,)
[[1 2 3]
 [4 5 6]]
(2, 3)
```

Prediction was perfect — (2, 3) exactly as you said.

---

But I need to flag something important from that output.

You ran the wrong Python.

Look at the first line of your terminal:
`c:/python314/python.exe c:/projects/myLLM/scratch.py`
That's Python 3.14 — the system Python, not your venv. This is the one that cannot run PyTorch with GPU support. It'll work fine for NumPy today, but the moment we touch PyTorch, it will either fail or run on CPU only.
This is the exact problem we fixed in Session 1.

Fix it now, before it becomes a habit.

In VS Code, look at the bottom right corner. It will show a Python version. Click it — a menu appears at the top of the screen listing all available Python installations.

You need to select the one that says something like:
`C:\projects\myLLM\venv\Scripts\python.exe`

Once you select it, the terminal prompt should show (venv) at the start.

Then run scratch.py again. The first line should change from `c:/python314/python.exe` to your venv path.

Tell me what it shows after you switch.

```
user@Shree-Ji-PC MINGW64 /c/projects/myLLM
$ C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe c:/projects/myLLM/scratch.py
Traceback (most recent call last):
File "c:\projects\myLLM\scratch.py", line 1, in <module>
import numpy as np
ModuleNotFoundError: No module named 'numpy'
(venv)
user@Shree-Ji-PC MINGW64 /c/projects/myLLM
```

Almost there — but not quite the right Python yet.

It switched to Python 3.11, which is correct. But it picked the system Python 3.11, not the one inside your venv. That's why NumPy isn't found — NumPy is installed in the venv, not system-wide.

Look at the path it used:
`C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe`
You need this one instead:
`C:/projects/myLLM/venv/Scripts/python.exe`

Fix it:

In VS Code bottom right, click the Python version again. In the list that appears, look specifically for the one that shows the venv folder inside your project. It might say something like:
`Python 3.11.9 ('venv': venv)`

or show the path:

`C:\projects\myLLM\venv\Scripts\python.exe`

If you don't see it in the list, do this in the VS Code terminal:
`C:\projects\myLLM\venv\Scripts\activate`

```
python scratch.py
```

That bypasses VS Code's selector entirely and runs directly inside the venv. Tell me what you get.wait everytime I'm selecting that file, I'm getting this notification "venv (3.11.9)" is already selected as the environment for: "myLLM"28 MarGood — that means VS Code knows about the venv. The selector is fine.

The issue is the terminal. VS Code has two ways to run a file and they're behaving differently. Let's ignore the play button for now and just use the terminal directly.

In the VS Code terminal, run these two commands one at a time:
`C:\projects\myLLM\venv\Scripts\activate`
Then:

```
python scratch.py
```

Tell me exactly what the prompt looks like after the first command — specifically whether you see (venv) at the start of the line.

    okay solved the play button actually now... we can continue.

Perfect. What did the output look like — did it show the venv path this time?

```
user@Shree-Ji-PC MINGW64 /c/projects/myLLM
$ c:/projects/myLLM/venv/Scripts/python.exe c:/projects/myLLM/scratch.py
[1 2 3]
(3,)
[[1 2 3]
 [4 5 6]]
(2, 3)
(venv)
```

That's the one. `c:/projects/myLLM/venv/Scripts/python.exe` — correct Python, correct environment. Everything from here runs properly.

---

**Exercise 2 — Matrix multiplication**

Add this to your scratch.py file below what you already have:

```python
a = np.array([[1, 2],
              [3, 4]])

b = np.array([[5, 6],
              [7, 8]])

c = np.matmul(a, b)
print(c)
print(c.shape)
```

Before running — two questions:

Will this multiplication work, or will NumPy reject it? Check the shapes.
What will c.shape be?

Think it through using what you know about matrix multiplication — the middle numbers must match. Tell me your predictions, then run it.

```
okay we have 2 matrix here and a 3rd matrix c...

1. a.shape = (2,2) and same for b.shape
2. the middle number matches, so the multiplication should work
3. and c.shape should also be (2,2).
4. and yes, the prediction is correct
```

Perfect. All four points correct, and you reasoned through it properly rather than just guessing.

---

Now let's make it more interesting.

**Exercise 3 — A shape that doesn't match**

Replace your a and b with these and try to multiply them:

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

b = np.array([[1, 2],
              [3, 4]])

c = np.matmul(a, b)
print(c)
print(c.shape)
```

Before running — three questions:

- What is a.shape?
- What is b.shape?
- Will this work or will NumPy throw an error? Use the middle numbers rule.

Tell me your predictions
first.a.shape is (2,3)
b.shape is (2,2)it should throw an error, the middle number does not match.

Exactly right. Run it and tell me what error NumPy gives you.Traceback (most recent call last):

```
File "c:\projects\myLLM\scratch.py", line 31, in <module>
c = np.matmul(a, b)
^^^^^^^^^^^^^^^
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 3)
```

Perfect. Read that error message — it's actually telling you exactly what you already knew.

"size 2 is different from 3" — the middle numbers don't match. a has 3 columns, b has 2 rows. NumPy caught it and refused.

That error message will become familiar. When you see matmul and mismatch in core dimension — you'll know immediately: shapes don't align. Check the middle numbers.

---

**Exercise 4 — Fix it yourself**

Same two arrays. Your job: change b so the multiplication works and c.shape comes out as (2, 4).

Don't change a. Only change b.
Tell me what shape b needs to be, write the new b, and run it.

```
user@Shree-Ji-PC MINGW64 /c/projects/myLLM
$ c:/projects/myLLM/venv/Scripts/python.exe c:/projects/myLLM/scratch.py
[1 2 3]
(3,)
[[1 2 3]
 [4 5 6]]
(2, 3)
[[19 16 27 32]
 [43 40 63 74]]
(2, 4)
```

That's exactly right. You didn't just run it — you figured out the shape first and built the array to match. That's the skill.

You've now done in actual code everything we covered conceptually last session:

`Arrays and shapes` — understood and verified

`Matrix multiplication` — predicted, confirmed, broke it, fixed it yourself

That's Phase 1 math and NumPy done in practice.

One question to close it out properly:

When you fixed b to make c.shape come out as (2, 4) — how did you reason about it? What shape did you decide b needed to be, and why?
Just so I capture your thinking in the documents accurately.

```
here's how I thought.

1. the shape is a is (2, 3). meaning 2 rows and 3 column
2. the current shape of b was (2,2). now the middle number must match, meaning I have to make a new row and then the shape would be (3, 2).
3. but the result needed was (2, 4)... so, I needed now 2 new columns... and I just added them in b
```

That's clean reasoning. You worked it out from first principles — middle numbers must match, outer numbers become the result shape. You didn't guess, you derived it.
Just to make it explicit so it sticks permanently:

```
The rule you just used:
(2, 3) × (3, 4) → (2, 4)
a b c

Middle numbers must match → 3 and 3 ✓
Result shape = outer numbers → 2 rows from a, 4 columns from b
```

You reasoned backwards from the result you wanted. That's exactly how you'll think about tensor shapes in the Transformer later — and it gets complicated there. The fact that you can do it now from scratch matters.
