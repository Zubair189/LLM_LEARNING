# What is vector search?

> Week 07 – Embeddings and Vector Databases · Question 9

## In One Line
Vector search means **finding the stored vectors that are closest to a query vector**, so you find texts with the most similar meaning.

## Simple Explanation
This question is about the general search operation; Week 08 Q9 shows how RAG uses it to fetch chunks for an LLM.

**How it works:**

1. **Ahead of time:** turn every document into an embedding and store it.
2. **At search time:** turn the user's query into an embedding with the **same model**.
3. **Compare** the query vector with stored vectors (using cosine similarity or distance).
4. **Return the top-k**, meaning the k best matches (for example the top 3).

This is also called **nearest-neighbour search**: you look for the "neighbours" closest to the query.

**Two ways to do the comparing:**
- **Exact (brute force):** compare with every single vector. Perfectly accurate, but slow when you have millions.
- **Approximate (ANN):** use a smart index, like HNSW, that only checks promising areas. Almost as accurate and much faster.

**Link to LLMs:** vector search is the "retrieval" part of RAG. It finds the chunks the LLM will read before answering.

## Real-Life Example
You want the nearest coffee shop. Brute force means measuring the distance to every coffee shop in the country. A smarter way is to look only in your neighbourhood first. Both give you a nearby coffee shop; one is much faster.

## Code Example
A brute-force search over three documents:

```python
# Documents with tiny made-up vectors: [login, money, holidays]
docs = {
    "Reset your password on the login page": [0.9, 0.1, 0.0],
    "Update your card under Payments":       [0.2, 0.9, 0.1],
    "The office is closed on holidays":      [0.0, 0.1, 0.9],
}
query_vec = [0.8, 0.3, 0.0]   # "I can't log in to my account"

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))   # similarity score

# Compare the query with EVERY document (brute force)
scores = [(round(dot(query_vec, v), 2), text) for text, v in docs.items()]
scores.sort(reverse=True)                     # highest score first

for score, text in scores[:2]:                # keep the top-2
    print(score, text)
# 0.75 Reset your password on the login page
# 0.43 Update your card under Payments
```

**What happened:** we scored every document against the query and kept the 2 best. The login document came first.

## Watch Out
- Vector search **always** returns something, even for an off-topic question. The "nearest" result is not always a good one, so use a minimum score.
- Use the **same embedding model** for documents and queries, or the scores are meaningless.

## Remember
- Vector search = find the closest vectors to the query.
- Exact search checks everything; approximate search (ANN) is faster at large scale.
- It is the retrieval step inside RAG.

## Related Questions
- [What is approximate nearest neighbor search?](../Week-07-Embeddings-and-Vector-Databases/14-What-is-approximate-nearest-neighbor-search.md)
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
- [How does vector similarity retrieval work?](../Week-08-Retrieval-Augmented-Generation-RAG/09-How-does-vector-similarity-retrieval-work.md)
- [What is hybrid search (keyword + vector)?](../Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)
