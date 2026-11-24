# What is FAISS?

> Week 07 – Embeddings and Vector Databases · Question 13

## In One Line
FAISS is a **very fast library from Meta for finding similar vectors**, but it is a search engine only, not a full database.

## Simple Explanation
FAISS stands for **Facebook AI Similarity Search**. It is free, written in C++ for speed, and you use it from Python.

**How you use it:**

1. **Create an index:** the structure that holds your vectors. You tell it the dimension, e.g. 384.
2. **Add vectors:** `index.add(vectors)`.
3. **Search:** `index.search(query, k)` gives back the k closest vectors as **scores** and **positions** (0, 1, 2…).

**Two kinds of index to know:**
- **Flat** (`IndexFlatL2`, `IndexFlatIP`): compares the query with every vector. Exact, fine for small data.
- **Approximate** (`IndexHNSWFlat`, `IndexIVFFlat`): skips most vectors, so it is much faster for millions of items, with a tiny risk of missing the best one.

**Library vs database:** FAISS only stores numbers. It does **not** keep your text, metadata or filters. You must keep a separate list that maps position 0, 1, 2… back to your text. A vector database (like Chroma) does all that for you.

**Link to LLMs:** FAISS is a good choice for experiments or when you want maximum speed. Many vector databases use similar ideas inside.

## Real-Life Example
FAISS is a powerful car **engine**: extremely fast, but you must build the car around it (seats, doors, steering). A vector database is the **complete car**.

## Code Example
Needs `pip install faiss-cpu numpy` (not run here, so the output is an example):

```python
import faiss
import numpy as np

texts = ["cats", "dogs", "cars"]                  # we keep the text ourselves
vecs = np.array([[1.0, 0.0],                      # tiny 2-number vectors
                 [0.9, 0.1],
                 [0.0, 1.0]], dtype="float32")    # FAISS needs float32

index = faiss.IndexFlatL2(2)                      # exact search, dimension 2
index.add(vecs)                                   # store the vectors
print(index.ntotal)                               # 3

query = np.array([[1.0, 0.05]], dtype="float32")  # something "cat-like"
distances, positions = index.search(query, 2)     # top-2 closest
print([texts[i] for i in positions[0]])           # e.g. ['cats', 'dogs']
```

**What happened:** FAISS found the two vectors closest to the query and gave back their positions. We used our own `texts` list to turn positions back into words.

## Watch Out
- FAISS needs **float32** numbers. NumPy uses float64 by default, so convert first.
- For cosine similarity, normalize your vectors (`faiss.normalize_L2`) and use an inner-product index like `IndexFlatIP`.

## Remember
- FAISS = fast similarity search library from Meta.
- Flat index = exact; HNSW/IVF = approximate and faster.
- It stores only vectors; you manage text and metadata yourself.

## Related Questions
- [What is approximate nearest neighbor search?](../Week-07-Embeddings-and-Vector-Databases/14-What-is-approximate-nearest-neighbor-search.md)
- [What is ChromaDB?](../Week-07-Embeddings-and-Vector-Databases/12-What-is-ChromaDB.md)
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
- [What is Euclidean distance?](../Week-07-Embeddings-and-Vector-Databases/08-What-is-Euclidean-distance.md)
