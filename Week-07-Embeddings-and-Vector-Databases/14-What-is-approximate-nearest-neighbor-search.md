# What is approximate nearest neighbor search?

> Week 07 – Embeddings and Vector Databases · Question 14

## Short Answer
Approximate nearest neighbor (ANN) search finds vectors that are *very likely* the closest to a query without comparing against every vector. It trades a small amount of accuracy (recall, e.g. 95–99%) for huge speedups, turning a linear scan over millions of vectors into a millisecond lookup. The most widely used ANN algorithm today is HNSW (graph-based); others include IVF (clustering), product quantization (compression), and DiskANN (SSD-based).

## Detailed Explanation
**Why approximate?** Exact kNN costs O(N·d) per query. With 10M vectors × 768 dims that's ~7.7 billion multiply-adds per query, too slow for interactive apps. ANN indexes visit only a small, promising subset.

**Main families:**

| Algorithm | Idea | Key knobs | Used in |
|---|---|---|---|
| **HNSW** (Hierarchical Navigable Small World) | Multi-layer graph; greedy hop from coarse top layer toward the query | `M` (links/node), `ef_construction`, `ef_search` | pgvector, Chroma, Qdrant, Weaviate, Milvus, FAISS |
| **IVF** (Inverted File) | k-means clusters; search only `nprobe` nearest clusters | `nlist`, `nprobe` | FAISS, Milvus, pgvector (`ivfflat`) |
| **PQ** (Product Quantization) | Compress vectors into short codes | sub-vectors, bits | FAISS IVFPQ, billion-scale |
| **DiskANN / Vamana** | Graph stored on SSD | graph degree, beam width | Large on-disk indexes |

**HNSW intuition:**
```
Layer 2:  A ─────────────── F            (few nodes, long jumps)
Layer 1:  A ──── C ──── F ──── H
Layer 0:  A─B─C─D─E─F─G─H─I─J            (all nodes, short links)
Query enters at top, greedily moves closer, drops down a layer, repeats.
```

**Measuring quality, recall@k:** the fraction of the true top-k (from brute force) that ANN returns. Tune `ef_search` / `nprobe` up for higher recall, down for speed.

| Setting | Recall@10 | Latency |
|---|---|---|
| low `ef_search` (16) | ~0.85 | fastest |
| medium (64) | ~0.97 | fast |
| high (256) | ~0.995 | slower |
*(illustrative numbers; measure on your data)*

**Trade-offs:** HNSW uses more memory (stores the graph) and is slower to build; IVF needs training and can lose recall at cluster boundaries; PQ saves memory but reduces precision. Filtering during ANN search (e.g. by tenant) can reduce recall, and modern engines handle this with filter-aware traversal.

## Example
**Analogy:** To find the nearest gas station, you don't measure the distance to every station in the country. You head toward the right city, then the right neighbourhood, then check a few streets. You'll almost always find the closest one, occasionally the second-closest.

```python
import faiss
import numpy as np

rng = np.random.default_rng(0)
d, n = 128, 50_000
xb = rng.random((n, d), dtype=np.float32)
xq = rng.random((100, d), dtype=np.float32)

exact = faiss.IndexFlatL2(d); exact.add(xb)
_, gt = exact.search(xq, 10)                      # ground truth

hnsw = faiss.IndexHNSWFlat(d, 32); hnsw.add(xb)
for ef in (16, 64, 256):
    hnsw.hnsw.efSearch = ef
    _, I = hnsw.search(xq, 10)
    recall = np.mean([len(set(I[i]) & set(gt[i])) / 10 for i in range(len(xq))])
    print(ef, round(recall, 3))
# approx (varies by machine/version):
# 16 0.7x
# 64 0.9x
# 256 0.99
```

## Common Mistakes / Misconceptions
- **Assuming ANN results are exact.** Always benchmark recall against brute force.
- **Using ANN on tiny datasets.** Under ~50–100k vectors, a flat index is often fast enough and exact.
- **Never tuning `ef_search`/`nprobe`.** Defaults may be too low for your recall target.
- **Blaming the embedding model** for misses that are actually caused by low ANN recall or aggressive filtering.

## Key Takeaways
- ANN trades a little recall for orders-of-magnitude speed.
- HNSW is the default in most vector DBs; IVF and PQ help at very large scale.
- Recall@k vs latency is tuned with `ef_search` (HNSW) or `nprobe` (IVF).
- Measure recall on your own data before going to production.

## Related Questions
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [What is FAISS?](../Week-07-Embeddings-and-Vector-Databases/13-What-is-FAISS.md)
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
