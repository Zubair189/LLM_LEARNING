# How does vector similarity retrieval work?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 9

## In One Line
It turns the question into an embedding, finds the stored chunks whose embeddings are **closest**, and passes those chunks to the LLM.

## Simple Explanation
Week 07 Q9 explains vector search in general; this question shows how RAG uses it to pick chunks for the LLM.

**Step by step:**

1. **Embed the question** with the **same model** that was used for the document chunks.
2. **Compare** the question's embedding with the stored chunk embeddings, usually with cosine similarity (a closeness score).
3. **Pick the top-k** closest chunks, e.g. the best 3.
4. **Apply rules:**
   - a **metadata filter** (e.g. only HR documents),
   - a **minimum score** (threshold), so weak matches are dropped.
5. **Return** the chunks with their text and labels, ready to put in the prompt.

```
"Can I bring my dog to work?" → embed → compare with chunks → top-k → LLM
```

**Why the minimum score matters:** vector search always finds a "closest" chunk, even for an off-topic question. With a threshold, an unrelated question gets **no** chunks, and the app can honestly say "I don't know".

**Known weak spots:** exact codes and names (fix: add keyword search), and the right chunk ranked a bit too low (fix: add a reranker).

**Link to LLMs:** the chunks found here are exactly what the LLM reads, so this step decides how good the answer can be.

## Real-Life Example
A librarian who knows every book's "topic fingerprint". You describe what you need, the librarian makes a fingerprint of your request, and brings the 2 closest books from the HR shelf only, skipping books that are barely related.

## Code Example
```python
# Chunks stored earlier, with made-up vectors [pets, parking]
chunks = [
    ("Dogs are allowed in the office on Fridays.", [0.9, 0.1]),
    ("No animals are allowed in the labs.",        [0.6, 0.0]),
    ("Parking permits are renewed in January.",    [0.0, 0.9]),
]

def retrieve(query_vec, k=2, threshold=0.3):
    scored = []
    for text, vec in chunks:
        score = sum(a * b for a, b in zip(query_vec, vec))   # dot product similarity
        if score >= threshold:                                # skip weak matches
            scored.append((round(score, 2), text))
    scored.sort(reverse=True)                                 # most similar first
    return scored[:k]                                         # top-k

print(retrieve([0.9, 0.1]))   # "Can I bring my dog to work?"
# [(0.82, 'Dogs are allowed in the office on Fridays.'), (0.54, 'No animals are allowed in the labs.')]

print(retrieve([0.1, 0.1]))   # "What's the capital of France?" (off-topic)
# []  <- nothing is close enough, so the app can say "I don't know"
```

**What happened:** the dog question found the two pet-related chunks. The off-topic question found nothing above the threshold.

## Watch Out
- The nearest chunk is **not always relevant**. Use a threshold or a reranker.
- Using a different model for the question than for the chunks gives meaningless scores.

## Remember
- Embed question (same model) → find closest chunks → filter → send to LLM.
- A minimum score lets the app say "I don't know".
- Add keyword search and reranking to cover weak spots.

## Related Questions
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [What is a retrieval pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
- [What is metadata filtering?](../Week-08-Retrieval-Augmented-Generation-RAG/13-What-is-metadata-filtering.md)
- [What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
