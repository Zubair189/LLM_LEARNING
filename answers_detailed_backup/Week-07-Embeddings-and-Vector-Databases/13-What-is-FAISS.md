# What is FAISS?

> Week 07 – Embeddings and Vector Databases · Question 13

## Short Answer
FAISS (Facebook AI Similarity Search, from Meta AI) is an open-source C++ library with Python bindings for fast similarity search and clustering of dense vectors. It provides many index types, from exact brute force (`IndexFlatL2`, `IndexFlatIP`) to approximate ones (`IndexHNSWFlat`, `IndexIVFFlat`, product quantization), with optional GPU acceleration. It is a *library, not a database*: it doesn't store your text or metadata, and persistence and filtering are up to you.

## Detailed Explanation
**Common index types:**

| Index | Type | Needs training? | Notes |
|---|---|---|---|
| `IndexFlatL2` | Exact, L2 | No | Brute force; perfect recall; baseline |
| `IndexFlatIP` | Exact, inner product | No | Use with normalized vectors for cosine |
| `IndexHNSWFlat(d, M)` | Graph ANN | No | Great speed/recall; tune `hnsw.efSearch` |
| `IndexIVFFlat(quantizer, d, nlist)` | Cluster ANN | Yes (`train`) | Search `nprobe` of `nlist` clusters |
| `IndexIVFPQ` | Clusters + compression | Yes | Billion-scale, low memory, lower recall |
| `IndexIDMap` | Wrapper | No | Attach your own int64 IDs |

**Workflow:**
```
vectors (float32, shape N×d) ─► [train if IVF/PQ] ─► index.add() ─► index.search(q, k) ─► (distances, ids)
                                                         │
                                         faiss.write_index / read_index for saving
```

**Practical rules:**
- Inputs must be `float32` NumPy arrays of shape `(n, d)`.
- For cosine similarity: `faiss.normalize_L2(x)` (in place) on both data and queries, then use an inner-product index.
- `search` returns two arrays: scores/distances and integer positions (or your IDs via `IndexIDMap`); map IDs back to text yourself.
- `IndexFlatL2` returns *squared* L2 distances.

**FAISS vs a vector DB:** FAISS is often the engine *inside* other tools and is ideal for offline experiments, research, batch jobs, or when you want full control and maximum speed. Choose a vector DB (Chroma, Qdrant, pgvector) when you need CRUD, metadata filtering, persistence, and multi-user serving.

## Example
**Analogy:** FAISS is a high-performance car engine. Extremely fast, but you need to build the car around it (storage, metadata, API). A vector database is the complete car.

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
docs = ["Python is a programming language",
        "Bananas are rich in potassium",
        "JavaScript runs in the browser",
        "Apples and oranges are fruits"]
xb = model.encode(docs).astype("float32")
faiss.normalize_L2(xb)                       # cosine via inner product

index = faiss.IndexFlatIP(xb.shape[1])       # 384
index.add(xb)
print(index.ntotal)                          # 4

xq = model.encode(["Which languages do web developers use?"]).astype("float32")
faiss.normalize_L2(xq)
scores, ids = index.search(xq, 2)
for s, i in zip(scores[0], ids[0]):
    print(f"{s:.2f}  {docs[i]}")
# approx:
# 0.4x  JavaScript runs in the browser
# 0.3x  Python is a programming language

# Approximate alternative for large N:
hnsw = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efSearch = 64
hnsw.add(xb)
faiss.write_index(hnsw, "docs.index")        # persist manually
```

## Common Mistakes / Misconceptions
- **Passing float64 arrays** (NumPy's default): FAISS requires float32.
- **Forgetting to normalize** when using `IndexFlatIP` for cosine similarity.
- **Skipping `train()`** for IVF/PQ indexes, which raises an error.
- **Expecting metadata filtering and text storage** like a database. Keep a parallel list/table mapping IDs to text and metadata.

## Key Takeaways
- FAISS is a fast vector similarity *library* from Meta, CPU and GPU.
- Flat indexes are exact; HNSW and IVF are approximate and scale further.
- Use float32, normalize for cosine, train IVF/PQ indexes.
- Pair it with your own storage, or use a vector DB for full database features.

## Related Questions
- [What is approximate nearest neighbor search?](../Week-07-Embeddings-and-Vector-Databases/14-What-is-approximate-nearest-neighbor-search.md)
- [What is ChromaDB?](../Week-07-Embeddings-and-Vector-Databases/12-What-is-ChromaDB.md)
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
- [What is Euclidean distance?](../Week-07-Embeddings-and-Vector-Databases/08-What-is-Euclidean-distance.md)
