# Explain embeddings and vector search.

> Week 12 – Final Project and Revision · Question 4

## Short Answer
An **embedding** is a fixed-length vector of numbers that represents the meaning of text (or images), produced by an embedding model so that similar meanings land close together in vector space. **Vector search** finds the stored vectors closest to a query vector, using a similarity measure such as cosine similarity, usually with an approximate nearest neighbor (ANN) index in a vector database. Together they enable semantic search, which is the "R" in RAG.

## Detailed Explanation

### End-to-end: indexing and querying

```
 INDEXING (offline)                                QUERY (online)
 ┌──────────┐  chunk  ┌────────┐ embed ┌────────┐   "how do I get my money back?"
 │ Documents│───────▶│ Chunks │──────▶│Vectors │            │ embed (same model!)
 └──────────┘        └────────┘       │+ meta  │            ▼
                                      └───┬────┘        q = [0.12, -0.40, ...]
                                          ▼                  │
                               ┌─────────────────────┐       │
                               │ Vector DB / index   │◀──────┘ ANN search (cosine)
                               │ (Chroma, FAISS, ...)│
                               └──────────┬──────────┘
                                          ▼
                     top-k: "Refund policy: 30 days..." (score 0.83)
                            "Returns process..."        (score 0.79)
                                          ▼
                            context for the LLM (RAG)
```

### Key concepts

| Concept | Meaning |
|---|---|
| Vector / dimension | A list of numbers, e.g. 384, 768, or 1536+ values per text |
| Sentence vs word embeddings | Sentence embeddings represent a whole passage — what RAG needs |
| Cosine similarity | `a·b / (‖a‖‖b‖)`; measures angle, ignores length; 1 = same direction |
| Euclidean distance | Straight-line distance; equivalent ranking to cosine for normalized vectors |
| Exact (brute force) search | Compare against every vector — accurate, slow at scale |
| ANN search | HNSW, IVF, etc. — trades a tiny bit of recall for large speedups |
| Vector database | Stores vectors + metadata, supports ANN, filtering, updates |

### Why it beats keyword search for meaning
"How do I get my money back?" shares no keywords with "Refund policy", but their embeddings are close. Keyword search (BM25) is still useful for exact terms like product codes, so production systems often use **hybrid search** (keyword + vector) plus a **reranker**.

### Practical rules
- Use the **same embedding model** for documents and queries.
- **Normalize** vectors if your index uses inner product as cosine.
- Store **metadata** (source, date, access level) to filter and cite.
- Re-embed everything if you change embedding models — vectors from different models are not comparable.

## Example
**Analogy:** A library where books are shelved by *topic* rather than title. Each book gets map coordinates (embedding). To answer a question, you compute the question's coordinates and walk to the nearest shelves (vector search). An ANN index is like a store directory that gets you to the right aisle quickly without checking every shelf.

**Python – embeddings + cosine search with sentence-transformers:**
```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim, runs locally
docs = [
    "Refund policy: customers may return items within 30 days.",
    "Our office is open Monday to Friday, 9am to 5pm.",
    "Shipping takes 3-5 business days.",
]
doc_vecs = model.encode(docs, normalize_embeddings=True)
q_vec = model.encode(["How do I get my money back?"], normalize_embeddings=True)[0]

scores = doc_vecs @ q_vec            # cosine because vectors are normalized
best = np.argsort(-scores)[:2]
for i in best:
    print(f"{scores[i]:.2f}  {docs[i]}")
# 0.5x  Refund policy: customers may return items within 30 days.
# 0.1x  Shipping takes 3-5 business days.      (illustrative scores)
print(doc_vecs.shape)  # (3, 384)
```
For large collections, replace the NumPy dot product with FAISS or ChromaDB.

## Common Mistakes / Misconceptions
- **Mixing embedding models** between indexing and querying — results become meaningless.
- **Treating similarity as truth** — the nearest chunk may be related but not actually answer the question.
- **Assuming the LLM itself does the search** — retrieval is a separate step your code runs.
- **Embedding huge documents whole** — meaning gets diluted; chunk first.

## Key Takeaways
- Embeddings map meaning to vectors; similar meaning → nearby vectors.
- Vector search ranks stored vectors by similarity to the query vector (cosine is most common).
- ANN indexes make search fast at millions of vectors; vector DBs add metadata and filtering.
- Hybrid search + reranking improves precision in production.
- Embedding + vector search is the retrieval engine of RAG.

## Related Questions
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
- [What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [What is approximate nearest neighbor search?](../Week-07-Embeddings-and-Vector-Databases/14-What-is-approximate-nearest-neighbor-search.md)
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
- [What is a vector?](../Week-03-Mathematics-and-Neural-Network-Foundations/02-What-is-a-vector.md)
- [What is metadata filtering?](../Week-08-Retrieval-Augmented-Generation-RAG/13-What-is-metadata-filtering.md)
- [Explain the complete RAG pipeline.](05-Explain-the-complete-RAG-pipeline.md)
