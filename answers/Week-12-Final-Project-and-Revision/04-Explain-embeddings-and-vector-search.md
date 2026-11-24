# Explain embeddings and vector search.

> Week 12 – Final Project and Revision · Question 4

## In One Line
An **embedding** turns text into a list of numbers that captures its meaning, and **vector search** finds the stored texts whose numbers are closest to your question's numbers.

## Simple Explanation
Computers can't compare *meaning* directly, but they can compare numbers. An **embedding model** reads a piece of text and outputs a **vector**, a fixed-length list of numbers (for example 384 of them). The key idea: **texts with similar meaning get similar vectors**.

So "How do I get my money back?" and "Refund policy" share no words, but their vectors are close. Keyword search would miss this match; vector search finds it.

There are two phases:

```
Indexing:  Documents → Split into chunks → Embed each chunk → Store in vector database
Searching: Question  → Embed (same model!) → Find closest vectors → Top matches → (to the LLM)
```

**Step by step:**

**Indexing (done once, ahead of time)**
1. **Split** your documents into small pieces called **chunks**, e.g. a few paragraphs each.
2. **Embed** each chunk with the embedding model to get its vector.
3. **Store** each vector in a **vector database** (such as ChromaDB or FAISS), together with **metadata**: extra labels like the file name, page or date.

**Searching (every time a user asks)**
4. **Embed the question** using the **same** embedding model.
5. **Compare** the question's vector with the stored vectors. The usual measure is **cosine similarity**, which checks whether two vectors point in the same direction. 1 means "same meaning", around 0 means "unrelated".
6. **Return the top matches** (the "top-k", e.g. the best 4 chunks), often with their scores.
7. In RAG, those chunks are given to the LLM as context to answer from.

**Making it fast:** comparing against every vector is fine for thousands of chunks, but slow for millions. Vector databases use **approximate nearest neighbour (ANN)** indexes. They find *almost* the best matches much faster by not checking every single vector.

**Making it better:**
- **Hybrid search:** combine vector search (meaning) with keyword search (exact words like product codes).
- **Reranking:** a second model re-orders the top results more carefully.
- **Metadata filters:** e.g. only search documents this user is allowed to see.

## Real-Life Example
A library that shelves books by **topic** instead of by title. Each book gets map coordinates based on what it's about. To answer a question, you work out the question's coordinates and walk to the nearest shelves. An ANN index is like a store directory that sends you to the right aisle without checking every shelf.

## Watch Out
- Always use the **same embedding model** for documents and questions. Vectors from different models can't be compared, and if you switch models you must re-embed everything.
- "Closest" doesn't always mean "correct". The nearest chunk may be related but still not answer the question.

## Remember
- Embedding = **meaning as numbers**; similar meaning → nearby vectors.
- Vector search finds the **closest** stored vectors, usually by cosine similarity.
- This is the **"R" (retrieval)** in RAG.

## Related Questions
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
- [What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
- [What is vector search?](../Week-07-Embeddings-and-Vector-Databases/09-What-is-vector-search.md)
- [What is approximate nearest neighbor search?](../Week-07-Embeddings-and-Vector-Databases/14-What-is-approximate-nearest-neighbor-search.md)
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
- [Explain the complete RAG pipeline.](05-Explain-the-complete-RAG-pipeline.md)
