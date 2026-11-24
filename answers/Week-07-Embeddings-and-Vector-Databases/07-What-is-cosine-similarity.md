# What is cosine similarity?

> Week 07 – Embeddings and Vector Databases · Question 7

## In One Line
Cosine similarity is a score from **−1 to 1** that tells you whether two vectors **point in the same direction**.

## Simple Explanation
Picture each vector as an arrow starting from the same point. Cosine similarity looks only at the **angle** between the two arrows, not at how long they are.

- Arrows point the same way → score **1** (very similar).
- Arrows at a right angle → score **0** (unrelated).
- Arrows point opposite ways → score **−1** (opposite).

**How to calculate it (3 steps):**

1. **Dot product:** multiply matching numbers and add them up.
2. **Lengths:** find the length of each vector (square each number, add, take the square root).
3. **Divide** the dot product by the two lengths multiplied together.

**Worked example** with A = [1, 2, 2] and B = [2, 1, 2]:
- Dot product: 1×2 + 2×1 + 2×2 = **8**
- Length of A = √(1+4+4) = **3**, length of B = √(4+1+4) = **3**
- Cosine = 8 ÷ (3×3) = **0.889**

**Why use it for text?** The length of an embedding can be affected by things that are not meaning. Looking only at direction removes that noise. Tip: if vectors are already scaled to length 1 ("normalized"), cosine similarity is just the dot product.

**Link to LLMs:** most RAG systems use cosine similarity to decide which chunks are closest to the user's question.

## Real-Life Example
Two people point at stars in the night sky. Cosine similarity asks "are they pointing at the same star?", not "whose arm is longer?".

## Code Example
```python
import math

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))        # step 1: dot product
    len_a = math.sqrt(sum(x * x for x in a))      # step 2: length of a
    len_b = math.sqrt(sum(y * y for y in b))      #         length of b
    return dot / (len_a * len_b)                  # step 3: divide

A = [1, 2, 2]
B = [2, 1, 2]
C = [2, 4, 4]      # same direction as A, just twice as long
D = [-1, -2, -2]   # opposite direction to A

print(round(cosine(A, B), 3))   # 0.889 -> similar
print(round(cosine(A, C), 3))   # 1.0   -> same direction
print(round(cosine(A, D), 3))   # -1.0  -> opposite
```

**What happened:** C is twice as long as A but points the same way, so it still scores a perfect 1.0. Length doesn't matter, only direction.

## Watch Out
- A score of 0.8 does **not** mean "80% the same". It is just a closeness score.
- Some databases use **cosine distance** = 1 − cosine similarity. There, **lower** means more similar.

## Remember
- Cosine similarity = dot product ÷ (length × length).
- It measures **direction**, not size. Range −1 to 1.
- It is the default way to compare text embeddings.

## Related Questions
- [What is Euclidean distance?](../Week-07-Embeddings-and-Vector-Databases/08-What-is-Euclidean-distance.md)
- [What is semantic similarity?](../Week-07-Embeddings-and-Vector-Databases/06-What-is-semantic-similarity.md)
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [How does vector similarity retrieval work?](../Week-08-Retrieval-Augmented-Generation-RAG/09-How-does-vector-similarity-retrieval-work.md)
