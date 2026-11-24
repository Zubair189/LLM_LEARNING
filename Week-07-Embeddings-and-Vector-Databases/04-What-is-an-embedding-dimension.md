# What is an embedding dimension?

> Week 07 – Embeddings and Vector Databases · Question 4

## Short Answer
The embedding dimension is the number of values in each embedding vector, fixed by the model: `all-MiniLM-L6-v2` produces 384 dimensions, `bge-large` 1024, and OpenAI `text-embedding-3-large` 3072. More dimensions can capture finer distinctions but cost more storage, memory, and search time. Every vector in a single index must have the same dimension, and your vector database must be configured to match it.

## Detailed Explanation
- **Fixed per model.** You don't choose it per sentence; the model architecture (hidden size plus any projection) decides it.
- **Trade-off:**

| Dimension | Example models | Pros | Cons |
|---|---|---|---|
| 384 | all-MiniLM-L6-v2, bge-small | Fast, small, runs on CPU | Less nuance |
| 768–1024 | all-mpnet-base-v2, bge-large, e5-large | Good quality/cost balance | 2–3x storage |
| 1536–3072 | OpenAI text-embedding-3, many API models | Highest quality on benchmarks | Expensive at scale |

- **Storage math** (float32 = 4 bytes):

```
bytes = num_vectors × dims × 4
1,000,000 × 384  × 4 ≈ 1.5 GB
1,000,000 × 3072 × 4 ≈ 12.3 GB
```

- **Shrinking vectors:**
  - *Matryoshka embeddings* (supported by OpenAI text-embedding-3, nomic, many newer models) are trained so you can truncate to the first 256/512 dims with modest quality loss.
  - *Quantization*: store as int8 or binary (pgvector `halfvec`/`bit`, FAISS PQ) to cut memory 4–32x.
  - *PCA* as a last resort.
- **Higher isn't automatically better.** A well-trained 768-dim model can beat a weak 1536-dim one. Check benchmarks like MTEB *and* test on your own data.
- **Curse of dimensionality:** in very high dimensions distances concentrate, which is one reason exact search gets expensive and we use ANN indexes like HNSW.

## Example
**Analogy:** Describing a wine. With 3 dimensions (red/white, sweet/dry, price) you can roughly group wines. With 50 (tannin, oak, acidity, region...) a sommelier can distinguish subtle differences, but the tasting notes take longer to write and compare.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
print(model.get_sentence_embedding_dimension())  # 384

vec = model.encode("Embedding dimensions are fixed by the model")
print(vec.shape, vec.dtype)                       # (384,) float32
print(vec.nbytes, "bytes per vector")             # 1536 bytes per vector

# Storage estimate for 1M chunks
n = 1_000_000
print(f"{n * vec.nbytes / 1e9:.2f} GB")           # 1.54 GB
```

When creating an index you must pass the dimension explicitly, e.g. `faiss.IndexFlatIP(384)` or pgvector `embedding vector(384)`. Inserting a 768-dim vector into it raises an error.

## Common Mistakes / Misconceptions
- **Dimension mismatch** after switching models: the index was built for 384 but the new model outputs 768. Re-embed and rebuild.
- **Confusing embedding dimension with context length.** Dimension = vector size; max sequence length (e.g. 256 tokens for MiniLM) = how much text the model reads before truncating.
- **Assuming more dims = better retrieval.** Evaluate on your own queries.
- **Ignoring memory budgets** until production; plan storage with the formula above.

## Key Takeaways
- Embedding dimension = length of each vector; set by the model.
- Higher dimensions trade cost for potential accuracy.
- Index, schema, and model dimensions must all match.
- Matryoshka truncation and quantization reduce cost with small quality loss.

## Related Questions
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
- [What is a vector representation?](../Week-07-Embeddings-and-Vector-Databases/03-What-is-a-vector-representation.md)
- [What is approximate nearest neighbor search?](../Week-07-Embeddings-and-Vector-Databases/14-What-is-approximate-nearest-neighbor-search.md)
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
