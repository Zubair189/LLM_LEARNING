# How does vector similarity retrieval work?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 9

## Short Answer
In a RAG system, vector similarity retrieval embeds the user's question with the same model used to embed the document chunks, searches the vector index for the top-k chunks whose vectors are closest (usually by cosine similarity), applies any metadata filters or score thresholds, and returns those chunks with their text and metadata so they can be inserted into the LLM's prompt. Its job is to find text that is *about the same thing* as the question, even when the wording differs.

## Detailed Explanation
**Scope note:** Week 07 Q9 ("What is vector search?") covers the underlying search operation: kNN, metrics, exact vs ANN indexes. This question is about how that operation is *used as the retriever in RAG*: what goes in, which knobs matter, and what comes out for the generator.

```
"Can I bring my dog to work?"
        │ embed (same model, same normalization, required query prefix if any)
        ▼
   q = [0.04, -0.12, ...]  (384-d)
        │ ANN search (HNSW) in vector DB, filter: {"dept": "HR"}
        ▼
   top-k by cosine:  0.71 "Pets policy: dogs allowed on Fridays..."   (page 12)
                     0.55 "Office rules: no animals in labs..."       (page 14)
                     0.18 "Parking permits..."     ← below threshold 0.3, dropped
        ▼
   chunks + metadata ──► context injection ──► LLM
```

**RAG-specific knobs:**

| Knob | Effect | Starting point |
|---|---|---|
| `k` (top-k) | More recall vs more noise/tokens | Retrieve 20–50 if reranking, else 3–8 |
| Score threshold | Avoid irrelevant context; enable "I don't know" | Calibrate on real queries |
| Metadata filter | Scope by tenant, permission, date, product | Always filter by permission |
| Query embedding format | Some models need `"query: "` / instruction prefixes | Check model card |
| Asymmetric models | Questions vs passages embedded differently | Use retrieval-trained models (BGE, E5, multi-qa-MiniLM) |
| MMR (max marginal relevance) | Diversify results, avoid near-duplicates | λ ≈ 0.5–0.7 |

**Known failure modes and fixes:**
- *Vocabulary mismatch on exact terms* (SKUs, error codes) → hybrid search with BM25.
- *Question ≠ answer phrasing* → HyDE (embed a hypothetical answer) or query rewriting.
- *Right chunk ranked 7th* → rerank the top-k with a cross-encoder.
- *Off-topic question still returns results* → threshold + "not found" handling.

## Example
**Analogy:** A librarian who knows every book's "topic fingerprint." You describe what you need, the librarian converts your request into the same fingerprint format and pulls the 3 closest books from the HR section only (filter), skipping any that are only loosely related (threshold).

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = [
    {"text": "Pets policy: dogs are allowed in the office on Fridays.", "dept": "HR"},
    {"text": "Office rules: no animals are permitted in laboratories.",  "dept": "HR"},
    {"text": "Parking permits are renewed every January.",              "dept": "Facilities"},
]
E = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

def retrieve(question, k=3, dept=None, threshold=0.3):
    q = model.encode(question, normalize_embeddings=True)
    scores = E @ q
    results = []
    for i in np.argsort(-scores):
        if dept and chunks[i]["dept"] != dept:
            continue
        if scores[i] < threshold:
            break
        results.append((round(float(scores[i]), 2), chunks[i]["text"]))
    return results[:k]

print(retrieve("Can I bring my dog to work?", dept="HR"))
# approx: [(0.6x, 'Pets policy: dogs are allowed ...'), (0.3x, 'Office rules: no animals ...')]
print(retrieve("What's the capital of France?"))
# likely []  -> nothing passes the threshold, so the app can answer "I don't know"
```

## Common Mistakes / Misconceptions
- **Assuming the nearest chunk is relevant.** Vector search always returns *something*; use thresholds or rerankers.
- **Skipping query prefixes/instructions** required by E5/BGE-style models.
- **Fixed k for every question.** Broad questions may need more chunks; specific ones fewer.
- **Filtering after retrieval with small k**, leaving zero results; filter inside the query.

## Key Takeaways
- Embed query (same model) → top-k nearest chunks → filter/threshold → pass to LLM.
- Retrieval quality hinges on model choice, chunking, k, thresholds, and filters.
- Add hybrid search, reranking, and query rewriting to cover vector search's blind spots.
- Evaluate with recall@k and context precision/recall.

## Related Questions
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [What is a retrieval pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
- [What is metadata filtering?](../Week-08-Retrieval-Augmented-Generation-RAG/13-What-is-metadata-filtering.md)
- [What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
