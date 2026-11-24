# What is Euclidean distance?

> Week 07 – Embeddings and Vector Databases · Question 8

## Short Answer
Euclidean distance is the straight-line ("as the crow flies") distance between two points: the square root of the sum of squared differences across all dimensions. Unlike cosine similarity, it is sensitive to vector length as well as direction, and *lower* means more similar (0 = identical). For normalized embeddings, Euclidean distance and cosine similarity produce the same ranking.

## Detailed Explanation
**Formula (L2 distance):**

```
d(A, B) = √( Σ (aᵢ − bᵢ)² )
```

```
  B •
    |\
    | \  d = straight line
    |  \
    •───• A
```

**Worked example** with A = [1, 2, 2], B = [2, 1, 2]:
- Differences: (1−2, 2−1, 2−2) = (−1, 1, 0)
- Squares: 1 + 1 + 0 = 2
- d = √2 ≈ **1.414**

Now take A and C = [2, 4, 4] (same direction, twice as long): d = √(1+4+4) = **3.0**, yet cosine similarity is 1.0. Euclidean says "far apart," cosine says "identical direction." That's the key difference.

**Link to cosine for unit vectors:**

```
‖A − B‖² = ‖A‖² + ‖B‖² − 2A·B = 2 − 2·cos(A, B)
```

Normalizing A and B above (divide by 3): cos = 0.889, so d² = 2 − 2(0.889) = 0.222, d ≈ **0.471**. Ranking by smallest d = ranking by largest cosine.

| | Cosine similarity | Euclidean (L2) distance |
|---|---|---|
| Range | −1 to 1 | 0 to ∞ |
| Better = | Higher | Lower |
| Sensitive to magnitude | No | Yes |
| Common default for | Text embeddings | Images, some older models, FAISS `IndexFlatL2` |

**Which to use?** Use whatever the embedding model was trained with (model cards say this). Most sentence-transformers models expect cosine; if you normalize, L2 works identically. Many vector DBs default to L2 (Chroma's default space is `"l2"`; FAISS `IndexFlatL2`), so set the metric explicitly.

Note: some libraries return *squared* L2 (FAISS, Chroma) to save a square root; rankings are unchanged.

## Example
**Analogy:** Cosine asks "are two hikers heading in the same direction?" Euclidean asks "how far apart are they standing right now?" Two hikers on the same trail, one 1 km in and one 10 km in, share direction but are far apart.

```python
import numpy as np

A = np.array([1, 2, 2], float)
B = np.array([2, 1, 2], float)
C = np.array([2, 4, 4], float)

print(round(np.linalg.norm(A - B), 3))   # 1.414
print(round(np.linalg.norm(A - C), 3))   # 3.0  (same direction, still "far")

An, Bn = A / np.linalg.norm(A), B / np.linalg.norm(B)
print(round(np.linalg.norm(An - Bn), 3)) # 0.471
print(round(np.sqrt(2 - 2 * (An @ Bn)), 3))  # 0.471  (matches the identity)
```

## Common Mistakes / Misconceptions
- **Sorting descending by distance.** Lower distance = more similar.
- **Mixing metrics**: building the index with L2 on unnormalized vectors from a cosine-trained model can hurt ranking.
- **Forgetting that FAISS/Chroma report squared L2.** Don't compare these numbers against unsquared thresholds.
- **Assuming Euclidean is "more accurate."** Neither is universally better; match the model.

## Key Takeaways
- Euclidean distance = straight-line distance; 0 means identical.
- It's sensitive to magnitude, unlike cosine.
- On normalized vectors, L2 and cosine give identical rankings (`d² = 2 − 2cos`).
- Always set the vector DB's metric explicitly to match your model.

## Related Questions
- [What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [What is FAISS?](../Week-07-Embeddings-and-Vector-Databases/13-What-is-FAISS.md)
