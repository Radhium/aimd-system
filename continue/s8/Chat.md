# Session type: LEARNING

`Project phase`: Phase 4 — Transformer architecture (just starting)
`Last session (Session 8)`: Phase 3 was completed. You built network3.py — a classification network with Softmax and cross-entropy loss, achieving 100% accuracy. Two permanent decisions were locked in: no Matplotlib, GPU enabled from here on.
`Today's goal`: Embeddings — how a token (a character or word piece) becomes a vector the Transformer can work with. This is the first concept of Phase 4.

---

Before I begin — one quick check.

When I say "token" today, I'll mean the smallest unit of text your model will process. For now we're thinking character-level (each letter is a token), which we decided in Session 1. Does that feel clear, or do you want me to briefly re-confirm what character-level means before we dive into what happens to a token once we have one?

```
it's clear, continue
```

**Good. Let's begin.**

**What is an embedding — and why do we need one?**

Your network from Phase 3 worked with numbers. You gave it `[2.0, 3.0]` and it did math on them. Clean and simple.
But text is not numbers. The letter `A` is not a number. The word `cat` is not a number. Before your Transformer can do anything at all, it needs to turn text into numbers — specifically into vectors.
That translation process is called `embedding.`

---

**Step 1 — First, give every token an integer ID**

Before embedding, we do something even simpler: we assign every character in our vocabulary a unique integer.

Imagine your dataset only contains these characters:

`a, b, c, d, e`

You make a lookup table:

```
a → 0
b → 1
c → 2
d → 3
e → 4
```

Now the word `bad` becomes `[1, 0, 3]`. This is called `tokenisation` — converting text into a list of integer IDs. You'll build the real tokeniser in Phase 5. For now, just know it exists and gives you a list of integers.

**Step 2 — The problem with raw integers**

You might think: great, a=0, b=1, c=2 — just feed those integers into the network directly.

The problem: integers carry accidental meaning.

If a=0 and e=4, the network might conclude that e is "four times more" than a, or that c sits exactly halfway between them. That is nonsense. These letters have no numeric relationship. The numbers are just ID tags — like locker numbers at school. Locker 50 is not "more" than Locker 1. They just happen to be labelled that way.

So we cannot feed raw integers into the network.

**Step 3 — The embedding table**

Here is the solution. We create a table — one row per token, one column per embedding dimension.

Imagine we have 5 tokens and we choose an embedding dimension of 4. The table looks like this:

```
Token I| Ddim 0| dim 1| dim 2| dim 3|
0 (a)| 0.21| −0.54| 0.87| 0.11|
1 (b)| −0.33| 0.72| 0.05| −0.91|
2 (c)| 0.61| 0.14| −0.22| 0.48|
3 (d)| −0.08| −0.67| 0.39| 0.75|
4 (e)| 0.44| 0.29| −0.81| −0.13|
```

Each row is a `vector` — a list of 4 numbers that represents that token.

When the model sees token ID `1` (the letter `b`), it looks up row 1 and gets `[−0.33, 0.72, 0.05, −0.91]`. That vector is what flows into the Transformer. The integer 1 never enters the network directly.
This lookup operation — "give me row N from this table" — is all an embedding is. It is a table lookup.

---

**Step 4 — The crucial part: these numbers are learned**
Those values in the table are not fixed. At the start of training, they are initialised randomly. During training, backpropagation adjusts them — exactly like it adjusts weights in your Phase 3 network.
By the end of training, tokens that appear in similar contexts end up with similar vectors. The model does not know what "similar" means in advance — it discovers it from the data.
This is the key insight:

```
An embedding is not a description of a token. It is a learned position in a high-dimensional space, shaped entirely by how the token behaves in context.
```

---

**Step 5 — What this looks like in PyTorch**
In PyTorch, the embedding table is one line:
`pythonnn.Embedding(vocab_size, embedding_dim)`

`vocab_size` — how many rows (one per token)
`embedding_dim` — how many columns (the size of each vector)

You pass in an integer token ID. You get back a vector. That's the whole operation.

In a real model with 50,000 tokens and an embedding dimension of 512, this table has 50,000 × 512 = `25.6 million parameters`. Embeddings are often one of the largest parts of the entire model.

**Warm-up check**

Before we move on — I want to make sure the core idea is solid.
Here is a small scenario. Don't calculate anything — just reason through it:

```
You have a vocabulary of 10 characters. Your embedding dimension is 6. You tokenise the word "ace" and get the IDs [0, 2, 4].
Question: What is the shape of the output after looking up all three token IDs in the embedding table? How did you get there?
```

Take your time — tell me your reasoning, not just the answer.

```
1. the shape should be (3, 6)
2. as for how we got, embedding table is a 10×6 matrix. Each row index corresponds to a token ID, and each row contains a 6-dimensional vector of floating-point numbers.
```

Both correct — and your reasoning is exactly right.
The word `"ace"` has 3 tokens. Each token retrieves one row of 6 numbers. So the output is 3 rows × 6 columns → shape `(3, 6)`. You got there by understanding the structure of the table, not by guessing. That is the right way to think about it.

---

**One more layer — what does this shape actually mean?**

That (3, 6) output has a name in Transformer terminology. It is called a `sequence of embeddings.`

1. The 3 is the sequence length — one position per token in your input
2. The 6 is the embedding dimension — the size of each token's vector

From this point forward, every component of the Transformer receives this shape as its input: `(sequence_length, embedding_dim)`. The attention mechanism, the feed-forward layers, everything — they all operate on this structure. Understanding it now will make Phase 4 much easier to follow.

**Where we go from here**
You now understand what an embedding is. The next concept builds directly on it.
Here is the problem we just created: after the embedding lookup, every position in the sequence holds a vector that describes what the token is — but nothing about where it appears in the sequence.
The vectors for "ace" and "eac" and "cea" would be identical sets of rows, just in a different order — but the embedding table has no way to encode that order. The Transformer, unlike an RNN, processes all positions at the same time. It has no built-in sense of left-to-right order.
This is the problem that `positional encoding` solves — and it is the next concept.
One question before we move on: does the idea that the Transformer has no built-in sense of order feel strange or surprising to you? I want to know if that needs more explanation before we go further, or if you're ready.

```
its a little non-sense, just somehow makes sense
```

That reaction is completely normal — and worth sitting with for a moment before we move on.

Here is why it feels like nonsense: every other sequence processing system you can imagine — reading, listening, typing — is fundamentally left to right. Order feels like it should be automatic.
But the Transformer was designed around one core operation: attention — which you haven't seen yet, but will shortly. Attention works by comparing every token to every other token simultaneously. It doesn't step through the sequence one position at a time. It sees the whole thing at once, like looking at a photograph rather than reading a book.
That design is what makes Transformers so fast and powerful. But it comes with a cost: the mechanism itself is order-blind. If you shuffle the tokens, the attention calculation produces the same result — just with shuffled rows.
So the engineers faced a choice: either redesign attention to be sequential (slow), or inject position information separately and keep attention fast. They chose the second option. That injection is positional encoding.
Think of it like this: imagine you receive a stack of index cards, each with a word on it, but no numbers on the back. You cannot tell which card came first. Positional encoding is the act of writing a number on the back of each card before handing the stack to the Transformer.

---

The "just somehow makes sense" feeling is actually the right place to be right now. You don't need it to feel natural yet — it will click properly once you see attention working. Positional encoding will make much more sense once you understand what it is feeding into.
Ready to move into positional encoding?

```
yes
```

---

**Positional Encoding**
Here is where we left off:

After the embedding lookup, you have a matrix of shape `(sequence_length, embedding_dim)`. Each row is a vector describing what a token is. Nothing in that matrix says anything about where each token sits in the sequence.

Positional encoding fixes this by adding a second matrix — same shape `(sequence_length, embedding_dim)` — where each row encodes the position of that token.
You then simply add the two matrices together, element by element:

```
token embeddings (what each token is)
+
positional encoding (where each token is)
─────────────────────────────────────────
input to Transformer (what + where, combined)
```

The result is still shape `(sequence_length, embedding_dim)`. Nothing about the shape changes. You have just baked position information into each vector.

**What does a position vector actually look like?**

There are two main approaches. I will explain both briefly, then tell you which one your model will use.

**Option 1 — Learned positional embeddings**

Exactly like the token embedding table, but for positions instead of tokens. You create a second table — one row per position, one column per embedding dimension. Position 0 gets a vector, position 1 gets a vector, and so on. These vectors are initialised randomly and learned during training, just like weights.

Simple. Flexible. Used by GPT-2 and most modern models.

The one limitation: if your table has 512 rows, the model has never seen position 513. It cannot generalise beyond the sequence length it was trained on.

**Option 2 — Fixed sinusoidal encoding**

The original Transformer paper used a mathematical formula — sine and cosine waves at different frequencies — to generate position vectors without any learning. The formula guarantees that every position gets a unique vector, and that the vectors have a smooth, consistent relationship to each other.

No parameters to learn. Works for any sequence length. But less flexible.

**Which one your model will use**

Your model will use `learned positional embeddings` — Option 1. Same reason we chose character-level tokenisation: simpler to implement, easier to understand, and used by the GPT family which is your reference architecture.
In PyTorch, it is literally the same line as your token embedding:

`pythonnn.Embedding(max_sequence_length, embedding_dim)`

Instead of passing a token ID, you pass a position index — `0, 1, 2, 3... `— and get back a vector for that position.

**The full picture so far**

Here is the complete input pipeline as it now stands:

```
Raw text: "ace"
Token IDs: [0, 2, 4]
Position IDs: [0, 1, 2]

Token embeddings: shape (3, 6) ← what each token is
Positional encoding: shape (3, 6) ← where each token sits
──────────────
Added together: shape (3, 6) ← ready for the Transformer

```

This combined matrix is the very first thing that enters the Transformer block. Everything that follows — attention, feed-forward layers, all of it — operates on this.

**Warm-up check**

Same style as before — reason through it, don't calculate:

```
Your model has a maximum sequence length of 128 and an embedding dimension of 64. You are processing a sequence of 10 tokens.
Question: What is the shape of the positional encoding matrix? And what is the shape of the final combined matrix after adding token embeddings and positional encoding together?
```

```
1. The shape of the full Positional Encoding lookup table is (128,64). as for this input, it should be (10, 64)
2. In the Transformer architecture, we do not append positional data; we add it element-wise to the token embeddings. For element-wise addition to occur, the two matrices must have identical dimensions.
 1.Token Embedding Matrix: You have 10 tokens, each with a 64-dimension vector. Shape: (10,64).
 2. Sliced Positional Encoding: You take the first 10 rows of the PE table. Shape: (10,64).
When you add these together:
(10,64)+(10,64)=(10,64)

The final combined matrix shape is (10,64).
```

That is a complete, precise answer — and you added something I did not explicitly tell you.

You worked out unprompted that you slice the first 10 rows from the full 128-row table, because the input is only 10 tokens long. That is exactly what happens in code. The table exists for up to 128 positions, but you only use as many rows as your current sequence needs.

Your explanation of why element-wise addition requires identical shapes is also exactly right — and that reasoning will keep coming back. Whenever two things are combined in a Transformer, shapes must be compatible. You now have the instinct to check that automatically.

**Where we are**

You have covered two concepts today:

`Embeddings` — a learned lookup table that turns token IDs into vectors. Shape out: (sequence_length, embedding_dim)
`Positional encoding` — a second learned lookup table that turns position indices into vectors. Added element-wise to the token embeddings. Shape unchanged.

Together these form the complete input to the Transformer. Every concept from here — attention, feed-forward layers, layer normalisation — receives this (sequence_length, embedding_dim) matrix and works on it.

**What comes next**

The next concept is `attention` — specifically, self-attention. This is the core mechanism of the Transformer. Everything else is built around it.

Here is a one-sentence preview so it lands better next session:

```
Self-attention is how each token in the sequence decides how much to pay attention to every other token when building its own representation.
```

That is enough for now. We will unpack it fully next session.
