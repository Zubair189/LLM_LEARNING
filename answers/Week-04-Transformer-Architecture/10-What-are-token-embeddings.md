# What are token embeddings?

> Week 04 – Transformer Architecture · Question 10

## In One Line
A token embedding is a **learned list of numbers for each token**, so the model can work with words as numbers where similar words get similar lists.

## Simple Explanation
A model can't read text directly. So the text goes through a few steps:

1. **Tokenizer:** split the text into **tokens** (words or word pieces) and give each one an ID number. `"The cat sat"` → `[464, 3797, 3332]`.
2. **Embedding lookup:** the model has a big table called the **embedding matrix**, with **one row per token** in its vocabulary. The ID simply says which row to read. That row is the token's embedding.
3. The embeddings (plus position information) go into the Transformer layers.

**Why not just use the ID number?** An ID like 3797 is just a label. It would wrongly suggest that token 3797 is "bigger" than token 12, and it says nothing about meaning.

**Why a list of numbers works:** the rows start random, but **training adjusts them**, like any other weight. Tokens used in similar situations end up with similar lists. "Cat" and "dog" land close together; "cat" and "sat" do not.

**Static vs. in context:** the starting embedding for "bank" is always the same. After the attention layers, the word's numbers change depending on context, so "river bank" and "bank loan" end up different.

**Link to LLMs:** the embedding table is the very first layer of every LLM. A big model may have over 100,000 tokens × 4,096 numbers each.

## Real-Life Example
Imagine a giant map where every word has a spot. "Cat" and "dog" live in the same neighbourhood (pets), while "sat" lives somewhere else (actions). The model learns where to put each word so the later layers can use the map.

## Code Example
```python
import math

vocab = {"the": 0, "cat": 1, "sat": 2, "dog": 3}   # token -> ID
E = [[0.1, 0.0, 0.2],     # row 0: the
     [0.9, 0.8, 0.1],     # row 1: cat
     [0.0, 0.3, 0.9],     # row 2: sat
     [0.8, 0.9, 0.2]]     # row 3: dog   (learned in a real model)

ids = [vocab[w] for w in ["the", "cat", "sat"]]
print(ids)                       # [0, 1, 2]
print([E[i] for i in ids])       # [[0.1, 0.0, 0.2], [0.9, 0.8, 0.1], [0.0, 0.3, 0.9]]

def similarity(a, b):            # cosine similarity: 1 = same direction
    dot = sum(x * y for x, y in zip(a, b))
    return round(dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))), 2)

print(similarity(E[1], E[3]), similarity(E[1], E[2]))   # 0.99 0.29
```

**What happened:** each ID picked one row of the table. "Cat" and "dog" have very similar rows (0.99), while "cat" and "sat" do not (0.29).

## Watch Out
- Embeddings are **learned**, not hand-made.
- A token is often a piece of a word (like "un"), not always a whole word.

Note: Week 7 Q1 covers embeddings in general (e.g. for search); this question is about the per-token lookup table at the start of a Transformer.

## Remember
- Token ID → row in the embedding table → a list of numbers.
- Similar tokens end up with similar embeddings.
- Context is added later by the attention layers.

## Related Questions
- [What is a token?](../Week-02-Introduction-to-LLMs-and-Generative-AI/06-What-is-a-token.md)
- [What are positional encodings?](../Week-04-Transformer-Architecture/09-What-are-positional-encodings.md)
- [What is a vector?](../Week-03-Mathematics-and-Neural-Network-Foundations/02-What-is-a-vector.md)
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
