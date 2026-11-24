# What is vector search?

> Week 07 – Embeddings and Vector Databases · Question 9

## Short Answer
Vector search finds the stored vectors closest to a query vector under a similarity metric (cosine, dot product, or Euclidean), i.e. a *k-nearest-neighbour* (kNN) lookup. For text, you embed the query with the same model used for the documents, then return the top-k nearest document vectors. At small scale this is an exact brute-force scan; at millions of vectors it uses approximate indexes like HNSW or IVF to answer in milliseconds.

## Detailed Explanation
**Scope note:** This question covers the *general search operation* (algorithms, metrics, exact vs approximate). Week 08 Q9, "How does vector similarity retrieval work?", covers how that operation is *used inside a RAG retriever* (query embedding, top-k, thresholds, filters, handing results to the LLM).

**Steps:**
```
index time:  docs ──embed──► vectors ──► index (flat / HNSW / IVF)
query time:  query ──embed──► q ──► find k nearest ──► ranked ids + scores
```

**Exact vs approximate:**

| Method | How | Complexity per query | Recall | Use when |
|---|---|---|---|---|
| Brute force (flat) | Compare q to every vector | O(N·d) | 100% | < ~100k vectors, or ground truth for testing |
| HNSW (graph) | Greedy walk through a layered proximity graph | ~O(log N) | ~95–99% | Default in most vector DBs, pgvector, Chroma |
| IVF (clusters) | Search only the `nprobe` closest clusters | ~O(N/nlist · nprobe) | Tunable | Very large, memory-sensitive sets |
| PQ / quantization | Compress vectors | Fast, small | Lower | Billions of vectors |

**Key knobs:** `k` (how many results), metric (match your model), and ANN parameters (HNSW `M`, `ef_search`; IVF `nlist`, `nprobe`) that trade recall for speed.

**Where it fits in 2026 stacks:** vector search alone is rarely the whole story. Production systems use **hybrid search** (vector + BM25), **metadata filters** (date, tenant, language), and a **reranker** on the top 20–100 candidates.

## Example
**Analogy:** Finding the nearest coffee shops on a map. Brute force measures the distance to every shop in the country; an ANN index is like checking only the neighbourhoods near you first.

Brute-force vector search in NumPy (what a flat index does):

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
docs = ["Reset your password from the login page",
        "Our office is closed on public holidays",
        "Update billing details under Account > Payments",
        "Two-factor authentication adds a security code"]
D = model.encode(docs, normalize_embeddings=True)          # (4, 384)

q = model.encode("I can't log in to my account", normalize_embeddings=True)
scores = D @ q                                              # cosine (unit vectors)
top_k = np.argsort(-scores)[:2]
for i in top_k:
    print(f"{scores[i]:.2f}  {docs[i]}")
# approx:
# 0.5x  Reset your password from the login page
# 0.3x  Two-factor authentication adds a security code
```

Swapping the NumPy scan for `faiss.IndexHNSWFlat(384, 32)` or a Chroma collection gives the same results approximately, at much larger scale.

## Common Mistakes / Misconceptions
- **Embedding the query with a different model** (or forgetting a required `"query: "` prefix) than the documents.
- **Assuming the top result is relevant.** Nearest ≠ correct; there is always a "nearest" even for off-topic queries. Use score thresholds or rerankers.
- **Using ANN for tiny datasets.** For a few thousand vectors, a flat scan is simpler and exact.
- **Not measuring recall** of the ANN index against brute force.

## Key Takeaways
- Vector search = kNN over embeddings with a chosen metric.
- Exact search is O(N); ANN indexes (HNSW, IVF) trade a little recall for big speedups.
- Same model and metric for queries and documents.
- Combine with keywords, filters, and reranking in production.

## Related Questions
- [What is approximate nearest neighbor search?](../Week-07-Embeddings-and-Vector-Databases/14-What-is-approximate-nearest-neighbor-search.md)
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
- [How does vector similarity retrieval work?](../Week-08-Retrieval-Augmented-Generation-RAG/09-How-does-vector-similarity-retrieval-work.md)
- [What is hybrid search (keyword + vector)?](../Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)
