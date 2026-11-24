# What is reranking and why does it improve RAG?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 16 · **BONUS**
>
> *Bonus question (added)*

## Short Answer
Reranking is a second retrieval stage: a fast first stage (vector, BM25, or hybrid search) fetches a broad set of candidates (e.g. top 50), then a more accurate but slower model, usually a **cross-encoder**, scores each (query, chunk) pair and reorders them, keeping only the best few (e.g. top 5) for the LLM. It improves RAG because the most relevant chunks end up in the prompt, which raises context precision, reduces noise and token cost, and leads to more faithful answers.

## Detailed Explanation
**Bi-encoder vs cross-encoder:**
```
Bi-encoder (embeddings, stage 1)          Cross-encoder (reranker, stage 2)
query ─► enc ─► q ┐                        [query ; chunk] ─► Transformer ─► relevance score
chunk ─► enc ─► d ┘─► cosine(q, d)         (reads both together, full attention between them)
fast: chunk vectors precomputed            slow: one model pass per pair, no precomputation
```

| | Bi-encoder retrieval | Cross-encoder reranker |
|---|---|---|
| Inputs | Query and doc encoded separately | Query + doc jointly |
| Speed | Milliseconds over millions | ~10–100 ms per batch of tens of candidates |
| Accuracy | Good recall, coarse ordering | Higher precision, fine-grained relevance |
| Role | Recall: "get it into the top 50" | Precision: "put it in the top 3" |

**Why it helps:**
- Embeddings compress a chunk into one vector, losing detail; cross-encoders see word-level interactions (negation, exact entities, whether the passage actually *answers* the question).
- LLMs perform worse with noisy context and may miss facts in the middle; fewer, better chunks help.
- Lets you retrieve wider (higher recall) without stuffing the prompt.

**Options (late 2026):** open models such as `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-v2-m3`, mxbai/Jina rerankers; hosted APIs such as Cohere Rerank and Voyage rerank; **LLM-based reranking** (ask an LLM to score/sort candidates: most accurate, slowest); and **ColBERT** late-interaction models as a middle ground.

**Typical numbers:** retrieve 20–100 → rerank → keep 3–10. Measure the gain with context precision, MRR, or nDCG before and after.

## Example
**Analogy:** A job search. An applicant-tracking system keyword-screens 1,000 CVs down to 50 (fast, rough). A hiring manager actually reads those 50 against the job description and picks the best 5 (slow, accurate). You'd never have the manager read all 1,000, and you'd never hire straight off the keyword screen.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

query = "Does the warranty cover water damage?"
docs = [
    "Water-resistant up to 50 metres for swimming.",
    "The warranty does not cover water damage or accidental drops.",
    "Our warranty lasts two years from purchase.",
    "Water damage claims must be filed within 14 days.",
]

# Stage 1: bi-encoder retrieval (fast)
bi = SentenceTransformer("all-MiniLM-L6-v2")
E = bi.encode(docs, normalize_embeddings=True)
q = bi.encode(query, normalize_embeddings=True)
stage1 = sorted(zip((E @ q).tolist(), docs), reverse=True)
print([d[:30] for _, d in stage1])
# order may put a "water"-heavy doc first, e.g. 'Water damage claims must be fi...'

# Stage 2: cross-encoder reranking (accurate)
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = ce.predict([(query, d) for _, d in stage1])
reranked = sorted(zip(scores, [d for _, d in stage1]), reverse=True)
print(reranked[0][1])
# likely: 'The warranty does not cover water damage or accidental drops.'
```

(sentence-transformers also provides `ce.rank(query, documents, top_k=3)` in recent versions.)

## Common Mistakes / Misconceptions
- **Reranking too few candidates.** If the right chunk isn't in stage 1's results, reranking can't recover it.
- **Running a cross-encoder over the whole corpus.** It's far too slow; it's only for a shortlist.
- **Treating reranker scores as calibrated probabilities** across queries; use them for ordering, and calibrate any threshold.
- **Ignoring latency budgets**: reranking adds tens to hundreds of ms; batch requests and keep the candidate count reasonable.

## Key Takeaways
- Reranking = cheap broad retrieval, then accurate cross-encoder re-scoring of the shortlist.
- Bi-encoders handle recall; cross-encoders handle precision.
- Better top-k → cleaner context → more faithful answers at lower token cost.
- Standard production pattern: hybrid retrieve → rerank → inject top 3–10.

## Related Questions
- [What is a retrieval pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
- [What is hybrid search (keyword + vector)?](../Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
- [What is semantic similarity?](../Week-07-Embeddings-and-Vector-Databases/06-What-is-semantic-similarity.md)
