# What is hybrid search (keyword + vector)?

> Week 07 – Embeddings and Vector Databases · Question 16 · **BONUS**
>
> *Bonus question (added)*

## Short Answer
Hybrid search runs a keyword (lexical) search, usually BM25, and a vector (semantic) search on the same query, then merges the two ranked lists into one. Keywords nail exact terms such as product codes, names, and error messages; vectors catch synonyms and paraphrases. Merging, often with Reciprocal Rank Fusion (RRF), gives better recall than either alone, which is why it's the default retrieval setup in most production RAG systems in 2026.

## Detailed Explanation
**Why each alone falls short:**

| Query | BM25 (keyword) | Vector |
|---|---|---|
| "error E1047 on startup" | Finds exact code | May return generic "startup errors" |
| "how do I get my money back" | Misses "refund policy" | Finds it |
| "iPhone 15 Pro battery" | Exact model match | May mix in iPhone 14 |

**Architecture:**
```
              ┌─► BM25 index ──► ranked list A ─┐
query ────────┤                                  ├─► fuse (RRF / weighted) ─► [rerank] ─► top-k
              └─► vector index ─► ranked list B ─┘
```

**Fusion methods:**
- **Reciprocal Rank Fusion (RRF)**: `score(d) = Σ 1 / (k + rank_i(d))`, with `k ≈ 60`. Uses only ranks, so it doesn't matter that BM25 scores (0–30+) and cosine scores (0–1) are on different scales. Simple and robust.
- **Weighted score fusion**: normalize both scores to 0–1, then `α·vector + (1−α)·keyword`. Tunable, but sensitive to normalization.

**Where you get it:** Elasticsearch/OpenSearch, Weaviate (`hybrid` with `alpha`), Qdrant (sparse + dense with fusion), Milvus, Pinecone (sparse-dense), Postgres (pgvector + `tsvector` full-text with RRF in SQL), and LangChain's `EnsembleRetriever`. **Learned sparse** models like SPLADE or BGE-M3 sparse output can replace BM25 for better term weighting.

**Typical pipeline:** retrieve ~50 from each side, fuse, rerank the top 20–50 with a cross-encoder, keep 3–8 for the LLM.

## Example
**Analogy:** Searching for a lost dog using both its tag number (exact match: fast, certain when it hits) and a description ("small brown terrier": catches it even without the tag). Combining both leads finds the dog more often.

RRF by hand (exact output):

```python
def rrf(ranked_lists, k=60):
    scores = {}
    for ranking in ranked_lists:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])

bm25_ranking   = ["d3", "d1", "d4"]
vector_ranking = ["d1", "d2", "d3"]
for doc, s in rrf([bm25_ranking, vector_ranking]):
    print(doc, round(s, 4))
# d1 0.0325   <- strong in both lists wins
# d3 0.0323
# d2 0.0161
# d4 0.0159
```

Full hybrid with real scorers:

```python
# pip install rank-bm25 sentence-transformers numpy
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

docs = ["Error E1047: license server unreachable at startup",
        "Refund policy: money back within 30 days",
        "Troubleshooting common startup problems",
        "Battery tips for iPhone 15 Pro"]
query = "E1047 when the app starts"

tok = lambda t: re.findall(r"\w+", t.lower())   # strips "E1047:" -> "e1047"
bm25 = BM25Okapi([tok(d) for d in docs])
kw_rank = list(np.argsort(-bm25.get_scores(tok(query))))

m = SentenceTransformer("all-MiniLM-L6-v2")
E = m.encode(docs, normalize_embeddings=True)
vec_rank = list(np.argsort(-(E @ m.encode(query, normalize_embeddings=True))))

for i, s in rrf([kw_rank, vec_rank])[:2]:
    print(round(s, 4), docs[i])
# expected: the E1047 doc first, then the generic startup doc
```

## Common Mistakes / Misconceptions
- **Adding raw BM25 and cosine scores.** Different scales; use RRF or normalize first.
- **Tokenization mismatches** in BM25 (case, punctuation, "E1047:" vs "E1047"); preprocess consistently.
- **Assuming hybrid always wins.** Usually it does, but measure recall@k on your own queries and tune weights.
- **Skipping reranking** after fusion; fused lists still benefit from a cross-encoder.

## Key Takeaways
- Hybrid = lexical (BM25/sparse) + semantic (dense) retrieval, fused.
- RRF is the simplest robust fusion: rank-based, scale-free, `k≈60`.
- It fixes vector search's weakness on exact identifiers and keyword search's weakness on paraphrases.
- Standard production pattern: hybrid retrieve → rerank → LLM.

## Related Questions
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [What is a vector representation?](../Week-07-Embeddings-and-Vector-Databases/03-What-is-a-vector-representation.md)
- [What is reranking and why does it improve RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/16-BONUS-What-is-reranking-and-why-does-it-improve-RAG.md)
- [What is a retrieval pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
