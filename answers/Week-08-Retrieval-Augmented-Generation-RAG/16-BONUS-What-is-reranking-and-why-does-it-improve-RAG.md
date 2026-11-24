# What is reranking and why does it improve RAG?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 16 · **BONUS**
>
> *Bonus question (added)*

## In One Line
Reranking is a **second, more careful check** that reorders the search results so the truly best chunks go to the LLM.

## Simple Explanation
Retrieval happens in two stages:

1. **Fast search:** vector (and/or keyword) search grabs a wide set of candidates, say the top 50. It is very fast because every chunk's embedding was computed ahead of time. But it is a bit rough: the question and each chunk were turned into numbers **separately**, so small details get lost.
2. **Reranking:** a slower but smarter model, usually a **cross-encoder**, reads the question and each candidate chunk **together** and gives a relevance score. We keep only the best few, say the top 5.

```
question → fast search (top 50) → reranker reads each pair → top 5 → LLM
```

**Why reading them together helps:** the reranker can notice details such as "does **not** cover", exact names, and whether the chunk actually **answers** the question rather than just sharing the topic.

**Why it improves RAG:**
- The best chunks end up at the top, so the LLM gets better evidence.
- Fewer, cleaner chunks mean a shorter prompt: cheaper, faster and less confusing.
- You can search wider in stage 1 without stuffing the prompt.

**Why not use the reranker on everything?** It's too slow. It must run once per (question, chunk) pair, so we only use it on a short list.

**Link to LLMs:** adding a reranker is one of the most effective upgrades for a basic RAG system.

## Real-Life Example
A job search. Software screens 1,000 CVs by keywords down to 50 (fast, rough). A hiring manager then reads those 50 carefully and picks the best 5 (slow, accurate). You'd never have the manager read all 1,000.

## Code Example
Needs `pip install sentence-transformers` (not run here, so the output is an example):

```python
from sentence_transformers import CrossEncoder

query = "Does the warranty cover water damage?"
candidates = [                                   # from stage 1 (fast search)
    "Water damage claims must be filed within 14 days.",
    "The warranty does not cover water damage or accidental drops.",
    "Our warranty lasts two years from purchase.",
]

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, c) for c in candidates])   # reads each pair together
best = sorted(zip(scores, candidates), reverse=True)          # highest score first
print(best[0][1])
# e.g. 'The warranty does not cover water damage or accidental drops.'
```

**What happened:** the reranker read the question with each chunk and moved the chunk that directly answers it to the top.

## Watch Out
- If the right chunk isn't in the stage-1 list, reranking can't find it. Give the reranker enough candidates (e.g. 20–50).
- Reranking adds some delay. Keep the candidate list a reasonable size.

## Remember
- Stage 1 = fast and wide; stage 2 (reranking) = slow and precise.
- A cross-encoder reads the question and chunk together.
- Better top chunks → better, cheaper answers.

## Related Questions
- [What is a retrieval pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
- [What is hybrid search (keyword + vector)?](../Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
- [What is semantic similarity?](../Week-07-Embeddings-and-Vector-Databases/06-What-is-semantic-similarity.md)
