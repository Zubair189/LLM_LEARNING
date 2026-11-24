# What is an embedding?

> Week 07 – Embeddings and Vector Databases · Question 1

## In One Line
An embedding is a **list of numbers that describes the meaning** of a piece of text, so that texts with similar meaning get similar numbers.

## Simple Explanation
Computers cannot understand words directly, but they are very good with numbers. An **embedding model** is an AI model that reads some text and turns it into a list of numbers, for example 384 numbers. That list is called an **embedding** (or a **vector**).

The magic is in how the numbers are chosen:

1. The model was trained on lots of text, so it learned which sentences mean similar things.
2. Sentences with **similar meaning** get **similar lists of numbers**. "How do I reset my password?" ends up close to "I forgot my login".
3. Sentences about different things, like "What's the weather in Paris?", end up far away.

A single number on its own means nothing to us. What matters is **which embeddings are close to each other**. Every text gets a list of the **same length**, whether it is 3 words or 300.

**Why is this useful?** Once meaning is numbers, we can use simple maths to find similar texts. This powers semantic search, recommendations and RAG.

**Link to LLMs:** In RAG, we turn documents and the user's question into embeddings, then hand the closest documents to the LLM so it can answer.

## Real-Life Example
Think of GPS coordinates. "48.85, 2.35" means little on its own, but two places with close coordinates are close in real life. Embeddings are like **GPS coordinates for meaning**.

## Code Example
Real embeddings come from a model and have hundreds of numbers. Here we use tiny made-up ones with 3 numbers.

```python
import math

# Tiny made-up embeddings (real ones have hundreds of numbers)
emb = {
    "reset my password": [0.9, 0.1, 0.2],
    "forgot my login":   [0.8, 0.2, 0.3],
    "weather in Paris":  [0.1, 0.9, 0.1],
}

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))      # multiply and add
    size_a = math.sqrt(sum(x * x for x in a))   # length of a
    size_b = math.sqrt(sum(y * y for y in b))   # length of b
    return dot / (size_a * size_b)              # score between -1 and 1

q = emb["reset my password"]
print(round(cosine(q, emb["forgot my login"]), 2))   # 0.98 -> very similar
print(round(cosine(q, emb["weather in Paris"]), 2))  # 0.24 -> not similar
```

**What happened:** we compared embeddings with **cosine similarity** (a closeness score). The two password sentences scored high; the weather one scored low.

In real code you would get embeddings from a model, e.g. `SentenceTransformer("all-MiniLM-L6-v2").encode("some text")`, which gives 384 numbers.

## Watch Out
- Always use the **same embedding model** for your documents and your questions. Embeddings from different models cannot be compared.
- Embeddings capture the general topic well, but they are weak at exact things like order numbers or error codes.

## Remember
- Embedding = a list of numbers that captures **meaning**.
- Similar meaning → **close** embeddings.
- Embeddings are the foundation of semantic search and RAG.

## Related Questions
- [What is a vector representation?](../Week-07-Embeddings-and-Vector-Databases/03-What-is-a-vector-representation.md)
- [What is an embedding dimension?](../Week-07-Embeddings-and-Vector-Databases/04-What-is-an-embedding-dimension.md)
- [What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
- [What is Retrieval-Augmented Generation (RAG)?](../Week-08-Retrieval-Augmented-Generation-RAG/01-What-is-Retrieval-Augmented-Generation-RAG.md)
