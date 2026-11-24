# What is cosine similarity?

> Week 07 – Embeddings and Vector Databases · Question 7

## Short Answer
Cosine similarity measures the angle between two vectors: it equals their dot product divided by the product of their lengths, giving a value from −1 (opposite) through 0 (unrelated/orthogonal) to 1 (same direction). Because it ignores vector length and looks only at direction, it's the standard metric for comparing text embeddings. If vectors are normalized to length 1, cosine similarity is simply the dot product.

## Detailed Explanation
**Formula:**

```
                 A · B            Σ aᵢbᵢ
cos(θ) = ───────────────── = ─────────────────────
           ‖A‖ × ‖B‖        √(Σ aᵢ²) × √(Σ bᵢ²)
```

```
        B
       /
      /  θ small → cos ≈ 1  (similar)
     /____________ A
```

| Value | Angle | Meaning for embeddings |
|---|---|---|
| 1.0 | 0° | Same direction (near-identical meaning) |
| ~0.5 | ~60° | Related |
| 0.0 | 90° | Unrelated |
| −1.0 | 180° | Opposite direction (rare for text models) |

**Worked example** with A = [1, 2, 2], B = [2, 1, 2]:
- Dot product: 1·2 + 2·1 + 2·2 = **8**
- ‖A‖ = √(1+4+4) = **3**, ‖B‖ = √(4+1+4) = **3**
- cos = 8 / (3×3) = **0.889**

Scaling doesn't matter: C = [2, 4, 4] = 2A gives cos(A, C) = **1.0**, even though C is "longer."

**Why it's popular for text:**
- Length of an embedding can reflect things like text length or word frequency, not meaning; cosine removes that.
- With normalized vectors, `cosine = dot product`, which is very fast and is what `IndexFlatIP` (FAISS) or `<#>`/`<=>` operators (pgvector) exploit.
- Relationship to Euclidean distance for unit vectors: `‖A−B‖² = 2 − 2·cos(A,B)`, so both give the same ranking.

**Cosine *distance*** = 1 − cosine similarity (used by Chroma's `"cosine"` space and pgvector's `<=>`); lower = more similar.

## Example
**Analogy:** Two people pointing at stars. Cosine similarity asks "are they pointing in the same direction?" not "how long are their arms?"

```python
import numpy as np

def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

A, B, C, D = [1, 2, 2], [2, 1, 2], [2, 4, 4], [-1, -2, -2]
print(round(cosine(A, B), 3))   # 0.889
print(round(cosine(A, C), 3))   # 1.0   (same direction, different length)
print(round(cosine(A, D), 3))   # -1.0  (opposite)
print(round(cosine([1, 0], [0, 1]), 3))  # 0.0 (orthogonal)

# With real embeddings
from sentence_transformers import SentenceTransformer, util
m = SentenceTransformer("all-MiniLM-L6-v2")
e = m.encode(["A cat sits on the mat", "A kitten rests on a rug"])
print(util.cos_sim(e[0], e[1]))  # approx tensor([[0.6x]])
```

## Common Mistakes / Misconceptions
- **Using dot product on unnormalized vectors and calling it cosine.** Normalize first (`normalize_embeddings=True`).
- **Reading cosine as a probability or percentage.** 0.8 doesn't mean "80% the same."
- **Confusing cosine similarity with cosine distance** in vector DB settings; one sorts descending, the other ascending.
- **Expecting negative scores** for "opposite" sentences; text embeddings usually cluster in a positive cone.

## Key Takeaways
- Cosine similarity = dot product ÷ product of lengths; range −1 to 1.
- It measures direction, not magnitude.
- For unit vectors: cosine = dot product, and ranking matches Euclidean distance.
- It's the default metric for most text embedding models.

## Related Questions
- [What is Euclidean distance?](../Week-07-Embeddings-and-Vector-Databases/08-What-is-Euclidean-distance.md)
- [What is semantic similarity?](../Week-07-Embeddings-and-Vector-Databases/06-What-is-semantic-similarity.md)
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [How does vector similarity retrieval work?](../Week-08-Retrieval-Augmented-Generation-RAG/09-How-does-vector-similarity-retrieval-work.md)
