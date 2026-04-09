# Session 10 — Self-Attention

Session type: LEARNING

Where we are
After Session 9, you know that every token enters the Transformer as a vector of numbers — its token embedding plus its positional encoding, giving a matrix of shape (seq_len, d_model).

The Transformer now has to do something with that matrix. It has to let tokens talk to each other.

**That is what self-attention does.**
The problem self-attention solves

Consider this sentence:

    "The cat sat on the mat because it was tired."

What does it refer to? The cat. Not the mat.

You know that immediately. But how? You looked at the other words in the sentence and decided which ones were relevant to understanding it. You attended to "cat" more than "mat."

A Transformer needs to do the same thing — for every token, figure out which other tokens are relevant to understanding it, and pull information from those tokens.

Self-attention is the mechanism that does this.

**The core idea — one sentence**
For every token, self-attention computes a weighted average of all the other tokens' values — where the weights come from how relevant each token is to the current one.

That's it. Everything else is just the machinery to compute those weights.

**The three vectors: Query, Key, Value**

Every token produces three vectors from its embedding. They are called:

    Query (Q) — "What am I looking for?"
    Key (K) — "What do I contain?"
    Value (V) — "What do I actually give out if someone attends to me?"

Here is the analogy that makes this concrete.
The library analogy

---

**Imagine a library.**

You walk in with a search query — say, "books about cats." That is your Query.

Every book on the shelf has a label on the spine — genre, topic, author. That label is the Key. It describes what the book contains.

You compare your query against every label. Some labels match well ("feline behaviour"), some match poorly ("accountancy").

From that comparison, you decide which books are worth pulling off the shelf.

When you pull a book off the shelf, what you actually read — the content inside — is the Value.

So:

    Query = what you're searching for
    Key = the label on each item (what it says it contains)
    Value = the actual content you receive

Every token plays all three roles simultaneously — it has its own Query (it searches), its own Key (it can be found), and its own Value (it can contribute content).

**How the weights are computed — step by step**

Let's say we have a sequence of 4 tokens. Each has been embedded into a vector.

**Step 1 — Create Q, K, V for every token**

Each token's embedding is multiplied by three separate learned weight matrices:

    WQ​ produces the Query vector
    WK​ produces the Key vector
    WV​ produces the Value vector

These matrices are learned during training. The network learns what to search for and what to advertise — not you.

**Step 2 — Compute attention scores**

For one token (say token 1), take its Query vector and compute a dot product with the Key vector of every token in the sequence (including itself).

The dot product measures similarity. A high score means "these two are relevant to each other." A low score means "not relevant."

You get one score per token. For 4 tokens: 4 scores.

**Step 3 — Scale the scores**

Divide every score by dkdk​
​, where dkdk​ is the dimension of the Key vectors.

Why? With large vector dimensions, dot products can get very large numbers. Large numbers fed into the next step cause gradients to vanish during training. Dividing by dkdk​
​ keeps the numbers in a stable range. This is just a stability trick — don't get stuck on it.

**Step 4 — Softmax**

Run the 4 scores through Softmax. You already know what Softmax does — it turns raw scores into probabilities that sum to 1.0.

Now you have 4 weights. They represent: how much attention should token 1 pay to each of the 4 tokens?

**Step 5 — Weighted sum of Values**

Multiply each token's Value vector by its attention weight, then sum them all up.

The result is a single vector — the new representation of token 1. It is a blend of information from all tokens, weighted by relevance.

This is done for every token in parallel. The output is a new matrix of the same shape as the input: `(seq_len, d_model).`

**The shape picture**
Thing | Shape
Input matrix | (seq_len, d_model)
Q matrix | (seq_len, d_k)
K matrix | (seq_len, d_k)
V matrix | (seq_len, d_v)
Attention scores | (seq_len, seq_len)
Output matrix | (seq_len, d_model)

The attention scores matrix is the interesting one. It is (seq_len, seq_len) — a grid where every row is one token asking "how much do I attend to each other token?" That grid is sometimes called the `attention map`.

**One thing to sit with**

Self-attention is position-agnostic by default. It doesn't know token 1 comes before token 2. It just sees vectors. That is exactly why positional encoding (Session 9) is needed — you bake the position information into the vectors before they enter attention.

**Warm-up check**

Before moving on, I want to make sure this has landed. Don't look back up — just reason through it.

Question: You have a sequence of 6 tokens. After computing Q, K, and V for all of them, you compute the attention scores. What shape is the attention scores matrix, and what does each row represent?

Take your time.

    the shape should be (6,6)
    The number of rows corresponds to the number of Query vectors

Both correct.

And you went one level deeper than I asked — you didn't just say "6 rows, one per token." You said Query vectors. That is the right way to think about it. Each row is one token asking its question against every other token's Key.

Let's go one step further into the Transformer before we close the session.

**Multi-head attention**

You now understand single-head attention. The real Transformer uses multi-head attention — and the idea is simple once you have the foundation.

**The problem with a single head**

A single attention head produces one attention map — one set of weights representing one way tokens relate to each other.

But tokens can relate to each other in multiple ways simultaneously.

In the sentence "John gave Mary the book because she asked him":

    she relates to Mary — coreference (who is she?)
    him relates to John — coreference (who is him?)
    gave relates to John and Mary — subject and object of an action
    book relates to gave — what was given

A single attention head can only capture one of these relationships at a time. It learns one WQ​, one WK​, one WV.

**The solution — run attention multiple times in parallel**

Multi-head attention runs h independent attention heads simultaneously. Each head has its own WQ, WK​, WV​ — its own learned way of deciding what is relevant.

One head might learn to track coreference. Another might track syntactic roles. Another might track proximity. The network decides — you don't specify it.

Each head produces its own output matrix of shape (seq_len, d_k).

**Combining the heads**

After all heads run, their outputs are concatenated side by side — stacked horizontally into one wide matrix.

Then that wide matrix is multiplied by one final learned weight matrix WOWO​ to project it back to shape `(seq_len, d_model).`

The output shape is the same as the input shape. The Transformer block doesn't change the dimensions — it just enriches the representations.

**The dimension split**

There is one practical detail worth knowing.

In GPT-style models, dk is set to d_model​÷h.

So if d_model = 64 and you have 8 heads, each head works in a space of dimension 8. The total computation stays the same size — you are splitting the model dimension across heads, not adding to it.

This is why the rule from Doc 2 says: number of attention heads must divide evenly into d_model.

**Second warm-up**

Same sequence — 6 tokens, d_model​ = 64, 4 attention heads.

Two questions:

    What is dkdk​ — the dimension each head works in?
    After all 4 heads run and their outputs are concatenated, what shape is the combined matrix before the final WOWO​ projection?

    dk is 64/4 = 16
    and shape of matrix is (6,64)

Both correct. And you got there cleanly — you didn't just divide, you tracked what concatenation does to the shape. That is the right instinct.

That is enough for one session. You have covered the two most important ideas in the Transformer:

    Self-attention — every token attends to every other token via Q, K, V
    Multi-head attention — run attention h times in parallel, each head learns a different relationship, concatenate and project back

---

**The Transformer block**
You now have attention. Inside every Transformer block there are two more components:

    The feed-forward layer
    Layer normalisation (applied twice — once after attention, once after feed-forward)

After these two, the full block is complete.

**The feed-forward layer**

After attention, each token has a new vector — a blend of information from the sequence. That vector now passes through a small fully-connected network.

This is called the `feed-forward layer` (sometimes written FFN or MLP).
What it looks like

Two linear layers with an activation function between them:

`x → Linear(d_model, 4 × d_model) → GELU → Linear(4 × d_model, d_model) → output`

The first layer expands the dimension by 4. The second layer compresses it back. The output is the same shape as the input.

This is sometimes called a bottleneck in reverse — you expand first, then compress.

**Why 4×?**

Empirically it works well. The expansion gives the network room to do more complex transformations before compressing back. You don't need a deeper reason than that — it is a hyperparameter that has become standard.

**Why does it exist at all?**

Attention is very good at moving information between tokens — deciding what to attend to. But it is not especially good at processing that information once it arrives.

The feed-forward layer is where each token processes what it collected from attention. It applies the same transformation independently to every token — it does not mix tokens together. Attention mixes, feed-forward processes.

A useful way to think about it: attention is communication, feed-forward is computation.

**GELU — the activation function**

You know ReLU from Phase 3 — max(0, z). In Transformers the standard activation is GELU (Gaussian Error Linear Unit).

The practical difference: ReLU is a hard threshold — it outputs exactly zero for anything negative. GELU is smooth — it allows small negative values through with a gentle curve rather than cutting them off sharply.

For Transformers, GELU trains slightly better. The intuition: a smooth function is easier for gradients to flow through. You don't need the formula — just know it is a smoother ReLU.

**Shape check**
Step | Shape
Input | (seq_len, d_model)
After first linear | (seq_len, 4 × d_model)
After GELU | (seq_len, 4 × d_model) — unchanged
After second linear | (seq_len, d_model)

In and out: (seq_len, d_model). The block never changes the shape.

**Warm-up**

d_model = 64. A sequence of 5 tokens passes through the feed-forward layer.

Two questions:

    What shape is the tensor after the first linear layer?
    What shape is it after the second linear layer?

    shape of tensor would be (5, 256) ig
    and final tensor would be (5, 64) ig

Both correct. And the "ig" is unnecessary — you derived them cleanly from first principles. Own it.

**Layer normalisation**

The last piece. This one is short.

**The problem it solves**

As data flows through a deep network, the numbers inside each layer can drift — some dimensions grow very large, others shrink toward zero. This makes training unstable. Gradients become unreliable. Learning slows or breaks.

You need a way to keep the numbers well-behaved at each step.

**What layer norm does**

For each token's vector independently, layer norm:

    1. Computes the mean of all values in that vector
    2. Computes the standard deviation of all values in that vector
    3. Subtracts the mean and divides by the standard deviation — this centres the values around zero with consistent spread
    4. Applies two learned scale parameters (called gamma and beta) that let the network rescale and shift the output if needed

The result: every token's vector has stable, normalised values — but the network can still learn to adjust the scale if that helps.

Shape in, same shape out. Layer norm never changes dimensions.

**Where it is applied**

In a GPT-style Transformer block, layer norm is applied before attention and before the feed-forward layer. This is called Pre-LN and is the modern standard. Some older diagrams show it after — don't let that confuse you.

So the full block order is:

```
input
→ Layer Norm
→ Multi-head attention
→ add input back (residual connection)
→ Layer Norm
→ Feed-forward layer
→ add input back (residual connection)
output
```

---

**The residual connection**

You just saw something new in that diagram — "add input back." This is called a residual connection (or skip connection).

After attention runs, instead of just passing the attention output forward, you add the original input back to it:

`output = attention(x) + x`

Same thing after the feed-forward layer:

`output = ffn(x) + x`

Why? Two reasons.

First, it gives gradients a clean highway back through the network during backprop — they can flow directly through the addition without passing through the attention or FFN computation. This prevents the vanishing gradient problem in deep networks.

Second, it means each block only needs to learn the change — the residual — rather than recomputing the full representation from scratch. That is much easier to learn.

This is why they are called residual connections — each block learns what to add, not what to replace.

**The full Transformer block, cleanly**

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Two lines. That is the entire block. You stack this N times (N = number of layers).

**Final warm-up for today**

No shape questions this time — a comprehension one.

    Question: In the residual connection x = x + Attention(LayerNorm(x)), why is it important that attention's output shape matches the input shape x?

Take your time.

    you say

I'll answer it — but I want to flag something first.

You've been getting every question right today. This one is answerable from what you already know. The shape rule, the addition operation — you have all the pieces.

"You say" is fine when something is genuinely unclear. But I don't think this one is unclear. I think you just didn't try.

So: one attempt. What happens when you try to add two matrices of different shapes?

    you cant add because of the miss match, it just would not make sense

Exactly. And that is the full answer.

The residual connection is an addition — x + Attention(LayerNorm(x)). Addition requires identical shapes. If attention changed the shape, the addition would be impossible. So attention must return (seq_len, d_model) — the same shape it received. Every component in the block is designed around this constraint.

You had that. Trust it next time.

**The full picture — what you now understand**

You have just completed the entire Transformer block. Let's put it together in one place.

```
Input: (seq_len, d_model)
  ↓
Token embeddings + Positional encoding       ← Session 9
  ↓
┌─────────────────────────────────┐
│  Transformer Block (× N layers) │
│                                 │
│  LayerNorm                      │
│  → Multi-head self-attention    │  ← Session 10
│  → + residual                   │
│                                 │
│  LayerNorm                      │
│  → Feed-forward (expand → GELU → compress) │
│  → + residual                   │
│                                 │
└─────────────────────────────────┘
  ↓
Final LayerNorm
  ↓
Linear layer → logits over vocab
  ↓
Output: (seq_len, vocab_size)
```

Every row in that diagram is something you now understand. Not just named — understood.

**What comes next**

Next session is BUILD. We write the Transformer block in code — model/transformer.py. You will see every concept from Sessions 9 and 10 appear as actual Python.
