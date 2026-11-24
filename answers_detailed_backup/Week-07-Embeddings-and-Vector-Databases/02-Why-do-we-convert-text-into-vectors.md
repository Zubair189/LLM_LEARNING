# Why do we convert text into vectors?

> Week 07 – Embeddings and Vector Databases · Question 2

## Short Answer
Computers can't do math on words, but they can do math on numbers. Converting text into vectors lets us *measure* how similar two pieces of text are (with cosine similarity or distance), search millions of documents by meaning in milliseconds, and feed text into machine-learning models. Keyword matching only finds the same words; vectors find the same *meaning*, so "car" can match "automobile."

## Detailed Explanation
Strings only support equality checks and substring matching. Vectors unlock a whole toolbox:

| Need | String approach | Vector approach |
|---|---|---|
| "Are these similar?" | Exact/fuzzy match | Cosine similarity score (−1 to 1) |
| Find related docs | `LIKE '%refund%'` | Nearest-neighbour search |
| Group topics | Manual tags | Clustering (k-means) |
| Feed to a classifier | Not possible directly | Vectors are model inputs |
| Cross-language match | Fails | Multilingual embeddings align languages |

**Why meaning beats keywords:**
- *Synonyms:* "cancel subscription" vs "stop my plan": zero shared keywords, high vector similarity.
- *Paraphrases:* "The bill was too high" vs "I was overcharged."
- *Ambiguity:* context-aware models place "bank (river)" and "bank (money)" in different regions.

**Why it matters for LLM apps:** LLMs have context-window limits and a knowledge cutoff. In RAG we embed a knowledge base once, then embed each user question and fetch the closest chunks to put in the prompt. Without vectors, there is no fast way to find "relevant by meaning."

**Caveat:** vectors are *worse* than keywords for exact tokens (order IDs, error codes, names). Production systems in 2026 typically combine both via **hybrid search** (BM25 + vectors), followed by **reranking**.

## Example
**Analogy:** A library sorted alphabetically by title (keywords) vs. one where a librarian shelves books by subject (vectors). In the second, walking to one book puts you next to related books even if their titles share no words.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
query = "cancel my subscription"
docs = ["How to stop your plan and end billing",
        "Subscription to our newsletter is free",
        "Best hiking trails in Colorado"]

# Keyword overlap: naive approach
q_words = set(query.split())
print([len(q_words & set(d.lower().split())) for d in docs])
# [0, 1, 0]  -> keyword match prefers the WRONG doc (newsletter)

# Vector approach
q = model.encode(query, normalize_embeddings=True)
D = model.encode(docs, normalize_embeddings=True)
print((D @ q).round(2))
# approx [0.5x 0.3x 0.0x] -> the "stop your plan" doc ranks first
```

## Common Mistakes / Misconceptions
- **"Vectors replace keyword search."** They complement it. Exact identifiers still need lexical (BM25) matching.
- **"Any numeric encoding works."** One-hot or word-count vectors don't capture meaning; you need learned embeddings for semantic similarity.
- **Ignoring preprocessing.** Embedding huge documents as a single vector blurs meaning; chunk first.
- **Forgetting cost/latency.** Embedding calls cost money/time; embed documents once at ingestion and cache.

## Key Takeaways
- Vectors let us compute similarity, which strings can't.
- Learned embeddings capture synonyms and paraphrases that keywords miss.
- Vectors enable fast semantic search at scale via vector indexes.
- Best practice: hybrid (keyword + vector) retrieval plus reranking.

## Related Questions
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
- [What is semantic similarity?](../Week-07-Embeddings-and-Vector-Databases/06-What-is-semantic-similarity.md)
- [What is hybrid search (keyword + vector)?](../Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)
- [Why do LLMs need external knowledge?](../Week-08-Retrieval-Augmented-Generation-RAG/02-Why-do-LLMs-need-external-knowledge.md)
