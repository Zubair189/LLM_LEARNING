# Week 07 – Embeddings and Vector Databases

This week covers how we turn text into numbers that capture meaning, and how we store and search those numbers at scale. You'll learn what embeddings are and how they differ from other vector representations, how embedding dimensions affect cost and quality, the difference between word and sentence embeddings, and how to measure similarity with cosine similarity and Euclidean distance. The second half moves from math to infrastructure: vector search, approximate nearest neighbour (ANN) indexes like HNSW and IVF, vector databases (ChromaDB, pgvector, Qdrant, Pinecone) versus libraries (FAISS), and how vector stores differ from relational databases. The week ends with an end-to-end recipe for retrieving similar documents, plus a bonus on hybrid (keyword + vector) search. Together these are the foundation for RAG in Week 08.

**Near-duplicate guide:** Q1 (embedding) is a *specific*, learned, dense kind of Q3 (vector representation), which is the umbrella idea that also covers one-hot and TF-IDF. Q9 (vector search) is the general kNN/ANN *operation*; Week 08 Q9 (vector similarity retrieval) is how that operation is *used as the retriever in RAG*. Q15 is the practical recipe that ties Q7–Q14 together.

## Questions
1. [What is an embedding?](01-What-is-an-embedding.md)
2. [Why do we convert text into vectors?](02-Why-do-we-convert-text-into-vectors.md)
3. [What is a vector representation?](03-What-is-a-vector-representation.md)
4. [What is an embedding dimension?](04-What-is-an-embedding-dimension.md)
5. [What is the difference between word and sentence embeddings?](05-What-is-the-difference-between-word-and-sentence-embeddings.md)
6. [What is semantic similarity?](06-What-is-semantic-similarity.md)
7. [What is cosine similarity?](07-What-is-cosine-similarity.md)
8. [What is Euclidean distance?](08-What-is-Euclidean-distance.md)
9. [What is vector search?](09-What-is-vector-search.md)
10. [What is a vector database?](10-What-is-a-vector-database.md)
11. [How is a vector database different from a relational database?](11-How-is-a-vector-database-different-from-a-relational-database.md)
12. [What is ChromaDB?](12-What-is-ChromaDB.md)
13. [What is FAISS?](13-What-is-FAISS.md)
14. [What is approximate nearest neighbor search?](14-What-is-approximate-nearest-neighbor-search.md)
15. [How do you retrieve similar documents using embeddings?](15-How-do-you-retrieve-similar-documents-using-embeddings.md)
16. **BONUS:** [What is hybrid search (keyword + vector)?](16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)

## Practical Exercise (1–2 hours): "Semantic FAQ Search, Three Ways"
Setup: `pip install sentence-transformers numpy faiss-cpu chromadb rank-bm25`

1. **Data (10 min):** write or collect ~40 FAQ entries (e.g. from a product help page you use). Include a few with exact codes or names ("Error E1047," "Plan Pro").
2. **Embeddings (15 min):** encode all entries with `all-MiniLM-L6-v2` (`normalize_embeddings=True`). Print the shape (expect `(40, 384)`) and compute cosine and Euclidean distance by hand with NumPy for two entries; confirm `d² = 2 − 2·cos`.
3. **Brute-force search (15 min):** write `search(query, k=3)` using `E @ q` and `argsort`.
4. **FAISS (15 min):** build `IndexFlatIP` and `IndexHNSWFlat(384, 32)`; confirm they return the same top-3 for 10 queries.
5. **ChromaDB (15 min):** load the same data into a `PersistentClient` collection with `{"hnsw:space": "cosine"}` and a `category` metadata field; run queries with a `where` filter.
6. **Hybrid (20 min):** add BM25 with `rank_bm25` and fuse with RRF (k=60).
7. **Evaluate (15 min):** write 10 test queries (half paraphrases, half exact codes) with the expected FAQ id. Compute hit@3 for vector-only, BM25-only, and hybrid. Write 3 sentences on what you observed.

**Deliverable:** one notebook/script plus a small table of hit@3 per method.

## Explain Aloud
Practise explaining each in under 60 seconds, without notes:
1. **Embeddings and why meaning becomes geometry:** what a 384-dim vector is, why similar sentences land close together, and why you must use the same model for queries and documents.
2. **Cosine similarity vs Euclidean distance:** the formulas, the worked example (A=[1,2,2], B=[2,1,2] → cos 0.889, L2 1.414), and why they rank identically on normalized vectors.
3. **Exact vs approximate nearest neighbour search:** why brute force doesn't scale, how HNSW navigates a layered graph, what recall@k means, and when a vector database (vs FAISS) is the right choice.
