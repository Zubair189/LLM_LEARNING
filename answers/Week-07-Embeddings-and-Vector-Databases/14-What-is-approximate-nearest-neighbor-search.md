# What is approximate nearest neighbor search?

> Week 07 – Embeddings and Vector Databases · Question 14

## In One Line
Approximate nearest neighbor (ANN) search finds the **almost-certainly closest** vectors **without checking every single one**, which makes search much faster.

## Simple Explanation
**The problem:** exact search compares the query with every stored vector. With 10 million vectors, that is 10 million comparisons for every question. Too slow for a chat app.

**The idea:** organise the vectors in advance so that, at search time, you only look in the **most promising places**. You might very rarely miss the true best match, but you get an answer in milliseconds. That small trade is the "approximate" part.

**Two common ways to organise vectors:**

1. **IVF (clusters):** group similar vectors into "neighbourhoods", each with a centre point. At search time, find the nearest centres, then search only inside those neighbourhoods.
2. **HNSW (graph):** connect each vector to its close neighbours, like a road network with highways on top. The search starts on the highways, jumps toward the query, then takes smaller roads to finish. HNSW is the most popular method today.

**How good is it?** We measure **recall**: out of the true top 10, how many did ANN find? Good settings often reach 95–99%. You can turn a setting up for better recall or down for more speed.

**Link to LLMs:** vector databases behind RAG (Chroma, pgvector, Qdrant…) use ANN, usually HNSW, so retrieval stays fast even with millions of chunks.

## Real-Life Example
To find the nearest petrol station, you don't measure the distance to every station in the country. You head to the right town, then the right neighbourhood, then check a few streets. You almost always find the closest one, and sometimes the second-closest.

## Code Example
A tiny "neighbourhood" search (the IVF idea):

```python
import math

# Vectors already sorted into 2 "neighbourhoods", each with a centre point
groups = {
    (1, 1): [[1, 2], [2, 1], [1, 1]],   # neighbourhood near (1, 1)
    (8, 8): [[8, 9], [9, 8], [7, 8]],   # neighbourhood near (8, 8)
}
query = [2, 3]

# Step 1: find the closest neighbourhood centre (only 2 checks)
centre = min(groups, key=lambda c: math.dist(c, query))

# Step 2: search only inside that neighbourhood (3 checks, not 6)
best = min(groups[centre], key=lambda v: math.dist(v, query))

print("searched near", centre)   # searched near (1, 1)
print("nearest:", best)          # nearest: [1, 2]
```

**What happened:** we skipped the whole far-away group and still found the closest vector. With millions of vectors, skipping like this saves a huge amount of time.

## Watch Out
- ANN results are **not guaranteed** to be the exact best. Check recall against an exact search on your own data.
- For small datasets (a few thousand vectors), exact search is fast enough and simpler.

## Remember
- ANN = skip most vectors, check only promising areas.
- Trades a tiny bit of accuracy for big speed.
- HNSW is the most common ANN method in vector databases.

## Related Questions
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [What is FAISS?](../Week-07-Embeddings-and-Vector-Databases/13-What-is-FAISS.md)
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
