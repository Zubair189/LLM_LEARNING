# What is hybrid search (keyword + vector)?

> Week 07 – Embeddings and Vector Databases · Question 16 · **BONUS**
>
> *Bonus question (added)*

## In One Line
Hybrid search runs a **keyword search** and a **vector (meaning) search** at the same time, then **merges** the two result lists into one.

## Simple Explanation
Each kind of search is good at different things:

- **Keyword search** (usually **BM25**, a classic method that scores documents by matching words) is great at **exact terms**: error codes like "E1047", product names, people's names.
- **Vector search** is great at **meaning**: it knows "how do I get my money back" matches "refund policy", even with no shared words.

Each one alone misses things the other would catch. So hybrid search does both:

1. Run keyword search → ranked list A.
2. Run vector search → ranked list B.
3. **Merge** the two lists into one final ranking.

```
question ─┬─► keyword search ─► list A ─┐
          └─► vector search  ─► list B ─┴─► merge ─► final results
```

**How to merge?** You can't just add the scores, because keyword scores and vector scores use different scales. A simple trick called **Reciprocal Rank Fusion (RRF)** looks only at each document's **position** in each list. A document gets `1 / (60 + position)` points from each list, and the points are added. Documents near the top of **both** lists win.

**Link to LLMs:** most production RAG systems use hybrid search, often followed by a reranker, so the LLM gets the best possible chunks.

## Real-Life Example
Looking for a lost dog using both its **tag number** (exact, certain when it matches) and a **description** ("small brown terrier", which works even without the tag). Using both clues, you find the dog more often.

## Code Example
```python
def rrf(ranked_lists, k=60):
    scores = {}
    for ranking in ranked_lists:
        for rank, doc in enumerate(ranking, start=1):
            # a doc gets more points the higher it is ranked
            scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])   # best first

keyword_ranking = ["d3", "d1", "d4"]   # from keyword search (BM25)
vector_ranking  = ["d1", "d2", "d3"]   # from vector search

for doc, score in rrf([keyword_ranking, vector_ranking]):
    print(doc, round(score, 4))
# d1 0.0325   <- high in BOTH lists, so it wins
# d3 0.0323
# d2 0.0161
# d4 0.0159
```

**What happened:** d1 and d3 appear in both lists, so they collect points twice and rise to the top. d2 and d4 appear only once.

## Watch Out
- Don't add raw keyword scores to cosine scores; they are on different scales. Use RRF (or scale both to 0–1 first).
- Hybrid usually helps, but test it on your own questions to be sure.

## Remember
- Hybrid = keyword search + vector search, merged.
- Keywords catch exact terms; vectors catch meaning.
- RRF merges lists using positions, not raw scores.

## Related Questions
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [What is a vector representation?](../Week-07-Embeddings-and-Vector-Databases/03-What-is-a-vector-representation.md)
- [What is reranking and why does it improve RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/16-BONUS-What-is-reranking-and-why-does-it-improve-RAG.md)
- [What is a retrieval pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
