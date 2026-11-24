# What is semantic similarity?

> Week 07 – Embeddings and Vector Databases · Question 6

## In One Line
Semantic similarity means **how close two texts are in meaning**, even if they use different words.

## Simple Explanation
"Semantic" just means "about meaning".

- "The flight was delayed" and "My plane left late" share almost no words, but they mean nearly the same thing. **High** semantic similarity.
- "Apple stock rose" and "apple pie recipe" share the word "apple", but they are about different things. **Low** semantic similarity.

**How do we measure it?**

1. Turn both texts into embeddings (lists of numbers that capture meaning).
2. Compare the two lists with a score, usually **cosine similarity**.
3. A higher score means closer meaning.

There is no universal "match" score. Every model gives slightly different numbers, so you choose a cut-off by testing on your own examples.

**Link to LLMs:** in RAG, we use semantic similarity to find the document chunks that are about the same thing as the user's question.

## Real-Life Example
Two friends describe the same movie in completely different words. You still realise they are talking about the same film. Recognising that is semantic similarity.

## Code Example
```python
import math

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

# Made-up 3-number meanings: [travel, food, money]
flight_delayed  = [0.9, 0.1, 0.2]
plane_left_late = [0.8, 0.1, 0.3]   # different words, same meaning
apple_pie       = [0.1, 0.9, 0.1]   # different meaning

print(round(cosine(flight_delayed, plane_left_late), 2))  # 0.99 -> very similar
print(round(cosine(flight_delayed, apple_pie), 2))        # 0.24 -> not similar
```

**What happened:** the two travel sentences point in nearly the same direction, so they score close to 1. The food sentence scores low.

## Watch Out
- **Similar is not the same as correct or relevant.** A passage can be on the same topic but not actually answer the question. This is why rerankers exist.
- Embeddings are weak at **negation and numbers**. "I love this phone" and "I don't love this phone" often score as very similar.

## Remember
- Semantic similarity = closeness of **meaning**, not of words.
- We measure it with embeddings + a score like cosine similarity.
- Pick your "good match" cut-off by testing, not guessing.

## Related Questions
- [What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
- [Why do we convert text into vectors?](../Week-07-Embeddings-and-Vector-Databases/02-Why-do-we-convert-text-into-vectors.md)
- [What is reranking and why does it improve RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/16-BONUS-What-is-reranking-and-why-does-it-improve-RAG.md)
