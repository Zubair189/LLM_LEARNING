# How do you retrieve similar documents using embeddings?

> Week 07 – Embeddings and Vector Databases · Question 15

## Short Answer
You (1) split documents into chunks, (2) embed each chunk with an embedding model, (3) store vectors plus text and metadata in a vector index or database, then at query time (4) embed the query with the *same* model, (5) run a top-k nearest-neighbour search (usually cosine), optionally with metadata filters, and (6) return the matching chunks ranked by score. Production systems often add hybrid keyword search and a reranker on the candidates.

## Detailed Explanation
```
INDEXING (offline, once per update)
 documents ─► clean ─► chunk ─► embed ─► store {id, vector, text, metadata}

QUERYING (online, per request)
 query ─► embed (same model) ─► top-k search (+ filters) ─► [rerank] ─► results
```

**Step-by-step decisions:**

| Step | Decision | Typical starting point |
|---|---|---|
| Chunking | Size and overlap | 200–500 tokens, 10–20% overlap |
| Embedding model | Quality vs cost, language | `all-MiniLM-L6-v2` to learn; BGE/E5/API models for production |
| Normalization / metric | Must match model | Normalize, cosine |
| Store | Library vs DB | NumPy/FAISS for experiments; Chroma/pgvector/Qdrant for apps |
| k | Recall vs noise | Retrieve 20–50, rerank, keep 3–8 |
| Filters | Scope results | `where={"lang": "en"}` |
| Threshold | Drop weak matches | Calibrate per model |

**Quality boosters (late-2026 practice):**
- **Hybrid search**: combine BM25 with vectors (e.g. Reciprocal Rank Fusion) to catch exact terms.
- **Reranking**: a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, bge-reranker, Cohere/Voyage rerank) rescores top candidates.
- **Query transformation**: rewrite or expand vague queries; HyDE (embed a hypothetical answer).
- **Evaluation**: build a small labelled set of (query, relevant doc) pairs and track recall@k and MRR.

This question is the *mechanics* of similarity retrieval; Week 08 places it inside a full RAG system.

## Example
**Analogy:** A music app's "songs like this." Every song gets a fingerprint once (embedding); when you pick a song, the app finds songs with the nearest fingerprints, optionally filtered to "only 2020s" (metadata), and then refines the order with a smarter model (reranker).

End-to-end in pure NumPy, no database:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
corpus = [
    {"text": "To reset your password, click 'Forgot password' on the login page.", "type": "faq"},
    {"text": "Our API rate limit is 100 requests per minute per key.",              "type": "docs"},
    {"text": "Invoices are emailed on the first day of each month.",                "type": "faq"},
    {"text": "Use exponential backoff when you receive HTTP 429 errors.",           "type": "docs"},
]

# 1-3. Embed and "store"
E = model.encode([d["text"] for d in corpus], normalize_embeddings=True)

def search(query, k=2, doc_type=None, min_score=0.2):
    q = model.encode(query, normalize_embeddings=True)       # 4. same model
    scores = E @ q                                             # 5. cosine
    order = np.argsort(-scores)
    hits = [(float(scores[i]), corpus[i]["text"]) for i in order
            if (doc_type is None or corpus[i]["type"] == doc_type)
            and scores[i] >= min_score]
    return hits[:k]                                            # 6. ranked results

for s, t in search("I'm getting too many requests errors", doc_type="docs"):
    print(f"{s:.2f}  {t}")
# approx:
# 0.4x  Use exponential backoff when you receive HTTP 429 errors.
# 0.4x  Our API rate limit is 100 requests per minute per key.
```

## Common Mistakes / Misconceptions
- **Different models or preprocessing for queries vs documents.** Vectors become incomparable.
- **Embedding whole documents** instead of chunks: meaning gets averaged out and text gets truncated.
- **Returning top-k blindly** with no threshold, so off-topic queries still get "results."
- **Not storing the original text/metadata** with the vector, so results can't be shown, cited, or filtered.

## Key Takeaways
- Chunk → embed → store → embed query → top-k search → return.
- Same model, same normalization, matching metric on both sides.
- Metadata filters and score thresholds keep results relevant.
- Hybrid search plus reranking is the standard upgrade path; measure with recall@k.

## Related Questions
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [What is ChromaDB?](../Week-07-Embeddings-and-Vector-Databases/12-What-is-ChromaDB.md)
- [What is a retrieval pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
- [What is hybrid search (keyword + vector)?](../Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)
