# What is a retrieval pipeline?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 8

## In One Line
A retrieval pipeline is the **series of steps that turns a question into a short list of the best matching chunks**, ready to give to the LLM.

## Simple Explanation
The retrieval pipeline is the "R" in RAG. It **stops before** the LLM writes anything. (The full **RAG pipeline**, Q11, is bigger: it also includes ingestion, building the prompt and generating the answer.)

A "pipeline" just means several steps in a row, where each step passes its result to the next:

1. **Prepare the question:** clean it up, maybe rewrite a vague question more clearly, then turn it into an embedding.
2. **Find candidates:** use vector search (and often keyword search too) to grab a wide set of possible chunks, e.g. the top 50.
3. **Filter:** remove chunks the user shouldn't see or that don't fit, using metadata (product, date, permissions).
4. **Rerank:** use a more careful model to reorder the candidates and keep only the best few, e.g. top 5.
5. **Tidy up:** drop weak matches below a minimum score and remove duplicates.

```
question → embed → find candidates → filter → rerank → top chunks → (to the LLM)
```

**Main idea:** first search **wide** (fast and cheap, so you don't miss anything), then **narrow down** (slower and more careful, so only the best survive).

**Link to LLMs:** most bad RAG answers come from bad retrieval, so this is where engineers spend much of their tuning time.

## Real-Life Example
Hiring. HR quickly scans hundreds of CVs for keywords (find candidates), removes people who don't meet the basic rules (filter), a manager reads the top 20 carefully (rerank), and 3 finalists meet the CEO (the LLM).

## Code Example
A tiny pipeline with filter, score, minimum score and top-k:

```python
# Chunks with made-up vectors [api_keys, cars] and metadata
chunks = [
    {"text": "API keys can be rotated in Settings.",   "vec": [0.9, 0.1], "product": "api"},
    {"text": "Rotate your tires every 10,000 km.",     "vec": [0.3, 0.9], "product": "auto"},
    {"text": "Revoked API keys stop working at once.", "vec": [0.7, 0.1], "product": "api"},
]

def retrieve(query_vec, product, k=2, min_score=0.3):
    results = []
    for c in chunks:
        if c["product"] != product:                                # 1. filter by metadata
            continue
        score = sum(a * b for a, b in zip(query_vec, c["vec"]))    # 2. similarity score
        if score >= min_score:                                     # 3. drop weak matches
            results.append((round(score, 2), c["text"]))
    results.sort(reverse=True)                                     # 4. best first
    return results[:k]                                             # 5. keep top-k

# Query "How do I rotate an API key?" already embedded as [1.0, 0.2]
print(retrieve([1.0, 0.2], product="api"))
# [(0.92, 'API keys can be rotated in Settings.'), (0.72, 'Revoked API keys stop working at once.')]
```

**What happened:** the "rotate your tires" chunk shares the word "rotate", but the product filter removed it. The two API chunks were scored, sorted and returned.

## Watch Out
- A retrieval pipeline is more than one search call. Filtering, reranking and minimum scores all matter.
- If the right chunk isn't among the first candidates, no later step can rescue it. Grab enough candidates.

## Remember
- Retrieval pipeline = question → candidates → filter → rerank → best chunks.
- It outputs chunks, not answers.
- Search wide first, then narrow down.

## Related Questions
- [What is a RAG pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/11-What-is-a-RAG-pipeline.md)
- [How does vector similarity retrieval work?](../Week-08-Retrieval-Augmented-Generation-RAG/09-How-does-vector-similarity-retrieval-work.md)
- [What is reranking and why does it improve RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/16-BONUS-What-is-reranking-and-why-does-it-improve-RAG.md)
- [What is hybrid search (keyword + vector)?](../Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)
