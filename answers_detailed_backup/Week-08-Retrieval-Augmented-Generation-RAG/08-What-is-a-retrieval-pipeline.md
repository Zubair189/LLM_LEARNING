# What is a retrieval pipeline?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 8

## Short Answer
A retrieval pipeline is the sequence of steps that turns a user's question into a short, ranked list of the most relevant chunks: query processing, candidate retrieval (vector, keyword, or hybrid), filtering, reranking, and post-processing (dedupe, threshold, top-k). It is the "R" in RAG and stops *before* the LLM writes the answer. Most RAG quality problems are retrieval problems, so this is where engineers spend much of their tuning effort.

## Detailed Explanation
**Scope note:** the *retrieval pipeline* (this question) ends with a list of chunks. The *RAG pipeline* (Q11) is the whole system: ingestion + retrieval pipeline + context injection + generation + citations/evaluation. Retrieval is a sub-pipeline of RAG.

```
question
   │
   ▼ 1. Query processing   rewrite / expand / HyDE / split multi-part questions
   ▼ 2. Candidate retrieval  vector top-50  +  BM25 top-50   → fuse (RRF)
   ▼ 3. Filtering          metadata: tenant, permissions, date, language
   ▼ 4. Reranking          cross-encoder rescoring → keep top 5
   ▼ 5. Post-processing    score threshold, dedupe overlaps, merge neighbours, order
   ▼
ranked chunks (+ metadata for citations)  ──► handed to generation
```

| Stage | Purpose | Example tools |
|---|---|---|
| Query processing | Make the query searchable | LLM rewrite, HyDE, multi-query |
| Retrieval | High-recall candidate set | FAISS, Chroma, pgvector, BM25, Elasticsearch |
| Filtering | Enforce scope and access | `where={...}` filters |
| Reranking | High-precision ordering | cross-encoders, Cohere/Voyage rerank, bge-reranker |
| Post-processing | Clean final context | dedupe, thresholds, max tokens |

**Design principle: recall first, precision second.** Retrieve wide (cheap, fast ANN/BM25), then narrow (expensive reranker on few candidates).

**Evaluate it separately** from generation: recall@k, MRR, nDCG on labelled queries, plus RAG metrics like **context precision** and **context recall** (Ragas).

## Example
**Analogy:** A hiring funnel. HR screens hundreds of CVs by keyword and quick match (retrieval), removes ineligible candidates (filtering), the hiring manager interviews the top 20 carefully (reranking), and 3 finalists go to the CEO (the LLM).

```python
import chromadb
from sentence_transformers import CrossEncoder

col = chromadb.Client().get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
col.add(ids=[str(i) for i in range(4)],
        documents=["API keys can be rotated in Settings > Security.",
                   "Rotate your tires every 10,000 km.",
                   "Revoked API keys stop working immediately.",
                   "Security settings include 2FA and SSO."],
        metadatas=[{"product": "api"}, {"product": "auto"},
                   {"product": "api"}, {"product": "api"}])

def retrieve(question, product, k_candidates=3, k_final=2):
    q = question.strip()                                            # 1. query processing
    res = col.query(query_texts=[q], n_results=k_candidates,        # 2. retrieval
                    where={"product": product})                     # 3. filtering
    cands = res["documents"][0]
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # 4. reranking
    scores = reranker.predict([(q, c) for c in cands])
    ranked = sorted(zip(scores, cands), reverse=True)
    return [c for s, c in ranked[:k_final]]                         # 5. top-k

print(retrieve("How do I rotate an API key?", product="api"))
# likely: ['API keys can be rotated in Settings > Security.', 'Revoked API keys stop working immediately.']
```

Note the "rotate your tires" chunk never appears: the metadata filter removed it before ranking.

## Common Mistakes / Misconceptions
- **Equating the retrieval pipeline with a single `similarity_search` call.** Real pipelines have several stages.
- **Reranking too few candidates** (if the right chunk isn't in the top-50, the reranker can't save it).
- **Only evaluating final answers.** Measure retrieval separately to know which stage failed.
- **Applying permission filters after generation** instead of during retrieval.

## Key Takeaways
- Retrieval pipeline = query processing → candidate retrieval → filtering → reranking → post-processing.
- It outputs ranked chunks, not answers; it's the R in RAG.
- Retrieve for recall, rerank for precision.
- Evaluate it with recall@k, MRR, and context precision/recall.

## Related Questions
- [What is a RAG pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/11-What-is-a-RAG-pipeline.md)
- [How does vector similarity retrieval work?](../Week-08-Retrieval-Augmented-Generation-RAG/09-How-does-vector-similarity-retrieval-work.md)
- [What is reranking and why does it improve RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/16-BONUS-What-is-reranking-and-why-does-it-improve-RAG.md)
- [What is hybrid search (keyword + vector)?](../Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)
