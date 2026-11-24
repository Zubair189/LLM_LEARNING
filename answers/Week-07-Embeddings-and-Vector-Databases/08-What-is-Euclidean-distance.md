# What is Euclidean distance?

> Week 07 – Embeddings and Vector Databases · Question 8

## In One Line
Euclidean distance is the **straight-line distance** between two points, like measuring with a ruler; **smaller means more similar**.

## Simple Explanation
Think of each vector as a point on a map. Euclidean distance is how far apart the two points are "as the crow flies".

**How to calculate it (3 steps):**

1. Subtract the numbers in matching positions.
2. Square each difference and add them up.
3. Take the square root.

**Worked example** with A = [1, 2, 2] and B = [2, 1, 2]:
- Differences: (−1, 1, 0)
- Squares added: 1 + 1 + 0 = 2
- Distance = √2 ≈ **1.414**

**How is it different from cosine similarity?**
- Cosine looks only at **direction** (higher = more similar).
- Euclidean looks at direction **and length** (lower = more similar, 0 = identical).

So C = [2, 4, 4] points the same way as A (cosine = 1.0) but is far away by Euclidean distance (3.0).

**Good news:** if all vectors are normalized (scaled to length 1), both methods give the **same ranking**. That is why many systems normalize embeddings first.

**Link to LLMs:** vector databases let you choose the distance type. Pick the one your embedding model expects (usually cosine) and set it explicitly.

## Real-Life Example
Two hikers on the same trail, one 1 km in and one 10 km in. They are walking in the **same direction** (cosine says "same"), but they are **9 km apart** (Euclidean says "far").

## Code Example
```python
import math

def euclidean(a, b):
    # square each difference, add them up, take the square root
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

A = [1, 2, 2]
B = [2, 1, 2]
C = [2, 4, 4]     # same direction as A, but twice as long

print(round(euclidean(A, B), 3))   # 1.414 -> fairly close
print(round(euclidean(A, C), 3))   # 3.0   -> "far", even though direction is the same
print(euclidean(A, A))             # 0.0   -> identical
```

**What happened:** B is close to A. C is far from A by distance, even though cosine similarity would call them a perfect match.

## Watch Out
- With distance, **lower is better**. Don't sort results from biggest to smallest by mistake.
- Some tools (like FAISS and Chroma) report the **squared** distance to save time. The ranking is the same, but the numbers look bigger.

## Remember
- Euclidean distance = straight-line distance; 0 means identical.
- It cares about length; cosine does not.
- On normalized vectors, both give the same ranking.

## Related Questions
- [What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [What is FAISS?](../Week-07-Embeddings-and-Vector-Databases/13-What-is-FAISS.md)
