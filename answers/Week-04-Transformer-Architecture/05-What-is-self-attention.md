# What is self-attention?

> Week 04 – Transformer Architecture · Question 5

## In One Line
Self-attention is attention where **every word in a sentence looks at all the other words in the same sentence** to understand its own meaning in context.

## Simple Explanation
In plain attention, the query can come from one place and the keys/values from another. In **self-attention**, they all come from the **same sentence**. The sentence "attends to itself".

**How it works:**
1. Each word starts as a list of numbers (its **embedding**).
2. From each word, the model makes a **query** (what it's looking for), a **key** (its label) and a **value** (its content).
3. Each word compares its query with **every** word's key, including its own, to get scores.
4. Softmax turns the scores into weights that add up to 1.
5. Each word's new version is a weighted mix of all the values.

All words do this **at the same time**, which is why Transformers are fast to train.

**Why it's powerful:** in "The animal didn't cross the street because **it** was too tired", the word "it" puts a lot of weight on "animal". Change "tired" to "wide", and "it" shifts towards "street". And the first and 1,000th words can connect in one step.

**Two flavours:**
- **Bidirectional** (encoders like BERT): each word sees words on both sides.
- **Causal / masked** (GPT-style decoders): each word sees only itself and earlier words.

**Link to LLMs:** every layer of an LLM runs (masked) self-attention, so each token's meaning gets refined by its context again and again.

## Real-Life Example
A group discussion where everyone listens to everyone else at once, paying more attention to the people most relevant to their own concern, and then updates their opinion. Everyone does this at the same time.

## Code Example
Three words, each a tiny 2-number vector. To keep it simple, each word's query, key and value are just its own vector.

```python
import math

words = ["cat", "dog", "pet"]
X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]   # "pet" shares features with both

for i, x in enumerate(X):
    scores = [sum(a * b for a, b in zip(x, y)) for y in X]   # compare with every word
    exps = [math.exp(s) for s in scores]
    weights = [round(e / sum(exps), 2) for e in exps]         # softmax
    print(words[i], weights)
# cat [0.42, 0.16, 0.42]     cat listens to itself and to "pet"
# dog [0.16, 0.42, 0.42]     dog listens to itself and to "pet"
# pet [0.21, 0.21, 0.58]     pet listens mostly to itself
```

**What happened:** each word got its own set of weights over all three words. In a real model, learned weight matrices make these scores reflect useful relationships, not just raw similarity.

## Watch Out
- "Self" does not mean a word looks only at itself; it looks at **all** words in its own sentence.
- Self-attention does not know word order on its own; that comes from positional encodings.

Note: Q4 covers attention in general; this question is the special case where Q, K and V all come from the same sentence.

## Remember
- Every word looks at every word in the same sentence.
- It gives each word a context-aware meaning.
- Encoders see both sides; GPT-style decoders see only the past.

## Related Questions
- [What is attention?](../Week-04-Transformer-Architecture/04-What-is-attention.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [What is multi-head attention?](../Week-04-Transformer-Architecture/08-What-is-multi-head-attention.md)
