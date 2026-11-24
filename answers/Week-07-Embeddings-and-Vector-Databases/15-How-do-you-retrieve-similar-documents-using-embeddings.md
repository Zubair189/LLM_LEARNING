# How do you retrieve similar documents using embeddings?

> Week 07 – Embeddings and Vector Databases · Question 15

## In One Line
Turn your documents and the question into embeddings with the **same model**, then return the documents whose embeddings are **closest** to the question's.

## Simple Explanation
The whole process has two parts.

**Part A – Prepare (done once, ahead of time):**

1. **Split** long documents into small pieces called **chunks** (a paragraph or so).
2. **Embed** each chunk: turn it into a list of numbers with an embedding model.
3. **Store** each embedding together with its text and metadata (labels like source or type).

**Part B – Search (every time someone asks):**

4. **Embed the question** with the **same** model.
5. **Compare** it with the stored embeddings (usually cosine similarity) and pick the **top-k** best. You can also filter by metadata, e.g. "only docs".
6. **Return** the matching chunks, best first.

```
documents → chunk → embed → store
question  → embed → compare → top-k chunks
```

**Easy upgrades later:** add keyword search (hybrid search) to catch exact codes, a **reranker** to reorder the top results more carefully, and a **minimum score** so off-topic questions get no results.

**Link to LLMs:** these steps are the retrieval half of RAG. Week 08 shows how the chunks are then handed to an LLM.

## Real-Life Example
A music app's "songs like this". Every song gets a fingerprint once (embedding). When you pick a song, the app finds songs with the nearest fingerprints, maybe only from the 2020s (filter).

## Code Example
```python
# Step 1-3: chunks, their (made-up) embeddings and metadata, stored together
store = [
    {"text": "Click 'Forgot password' to reset it.", "vec": [0.9, 0.1, 0.0], "type": "faq"},
    {"text": "API limit is 100 requests per minute.", "vec": [0.1, 0.9, 0.2], "type": "docs"},
    {"text": "Invoices are emailed monthly.",         "vec": [0.1, 0.2, 0.9], "type": "faq"},
    {"text": "Wait and retry after a 429 error.",     "vec": [0.2, 0.8, 0.1], "type": "docs"},
]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def search(query_vec, k=2, doc_type=None):
    hits = []
    for item in store:
        if doc_type and item["type"] != doc_type:      # optional filter
            continue
        hits.append((round(dot(query_vec, item["vec"]), 2), item["text"]))
    hits.sort(reverse=True)                            # best score first
    return hits[:k]                                    # top-k

# Step 4-6: the query "too many requests error" embedded by the SAME model
for score, text in search([0.1, 0.9, 0.1], doc_type="docs"):
    print(score, text)
# 0.84 API limit is 100 requests per minute.
# 0.75 Wait and retry after a 429 error.
```

**What happened:** we filtered to "docs" only, scored each chunk against the question, and returned the two best matches.

## Watch Out
- Using a **different model** (or settings) for the question and the documents breaks everything. The numbers no longer mean the same thing.
- Don't embed a whole long document as one vector. Its meaning gets blurred and the text may be cut off. Chunk first.

## Remember
- Chunk → embed → store, then embed question → compare → top-k.
- Same model on both sides.
- Keep text and metadata with each vector so you can show and filter results.

## Related Questions
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [What is ChromaDB?](../Week-07-Embeddings-and-Vector-Databases/12-What-is-ChromaDB.md)
- [What is a retrieval pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
- [What is hybrid search (keyword + vector)?](../Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)
