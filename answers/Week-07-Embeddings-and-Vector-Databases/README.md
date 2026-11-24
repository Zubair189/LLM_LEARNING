# Week 07 – Embeddings and Vector Databases

This week is about turning text into numbers that capture **meaning**, and then storing and searching those numbers quickly. First you'll learn what embeddings are, why we turn text into vectors, what "dimension" means, and the difference between word and sentence embeddings. Then you'll learn two simple ways to compare vectors: cosine similarity and Euclidean distance. The second half is about tools: vector search, fast approximate search (ANN), vector databases like ChromaDB, the FAISS library, and how vector databases differ from normal (relational) databases. The week ends with a step-by-step recipe for finding similar documents, plus a bonus on hybrid (keyword + vector) search. All of this is the foundation for RAG in Week 08.

**Similar-looking questions:** Q1 (embedding) is one special kind of Q3 (vector representation), the bigger umbrella idea. Q9 (vector search) is the general search operation; Week 08 Q9 shows how RAG uses it. Q15 puts Q7–Q14 together into one practical recipe.

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
2. **Embeddings (15 min):** turn all entries into embeddings with `all-MiniLM-L6-v2` (`normalize_embeddings=True`). Print the shape (expect `(40, 384)`), then compute cosine similarity and Euclidean distance by hand for two entries.
3. **Simple search (15 min):** write `search(query, k=3)` that scores every entry and returns the 3 best.
4. **FAISS (15 min):** build an exact index (`IndexFlatIP`) and a fast approximate one (`IndexHNSWFlat(384, 32)`); check that they give the same top 3 for 10 questions.
5. **ChromaDB (15 min):** load the same data into a `PersistentClient` collection with a `category` metadata field, and run questions with a `where` filter.
6. **Hybrid (20 min):** add keyword search with `rank_bm25` and merge the two lists with RRF (k=60).
7. **Evaluate (15 min):** write 10 test questions (half reworded, half with exact codes) and note the right FAQ id for each. Count how often the right answer is in the top 3 ("hit@3") for vector-only, keyword-only and hybrid. Write 3 sentences on what you noticed.

**Deliverable:** one notebook or script, plus a small table of hit@3 for each method.

## Explain Aloud
Practise explaining each one in under 60 seconds, in simple words, without notes:
1. **Embeddings:** what a list of 384 numbers is, why similar sentences get close numbers, and why questions and documents must use the same model.
2. **Cosine similarity vs Euclidean distance:** how to calculate each, the worked example (A=[1,2,2], B=[2,1,2] → cosine 0.889, distance 1.414), and why they give the same ranking on normalized vectors.
3. **Exact vs approximate search:** why checking every vector is too slow at scale, the basic idea of HNSW, what recall means, and when to pick a vector database instead of FAISS.
