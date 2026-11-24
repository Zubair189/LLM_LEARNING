# What is a vector database?

> Week 07 – Embeddings and Vector Databases · Question 10

## Short Answer
A vector database stores embeddings alongside their original content and metadata, and is optimized to answer "find the k most similar vectors to this one" quickly using ANN indexes such as HNSW. Beyond raw search (which a library like FAISS provides), it adds database features: persistence, CRUD updates, metadata filtering, scaling, access control, and often hybrid keyword search. Examples include Chroma, Pinecone, Weaviate, Qdrant, Milvus, and PostgreSQL with the pgvector extension.

## Detailed Explanation
**What a record looks like:**

```
id:        "faq-042"
vector:    [0.021, -0.113, ..., 0.067]     (384 floats)
document:  "Refunds are processed within 5 business days."
metadata:  {"source": "faq.pdf", "page": 3, "lang": "en", "updated": "2026-09-01"}
```

**Core capabilities:**
- **ANN indexing** (HNSW most common; IVF, DiskANN, quantization for scale).
- **Metadata filtering**: `where={"lang": "en"}` combined with similarity search.
- **CRUD + persistence**: add, update, delete vectors without full rebuilds; survive restarts.
- **Hybrid search**: many (Weaviate, Qdrant, Elasticsearch/OpenSearch, Milvus, pgvector + full-text) combine BM25 and vectors.
- **Ops features**: replication, sharding, multi-tenancy, backups, RBAC.

**Landscape (late 2026):**

| Type | Examples | Good for |
|---|---|---|
| Embedded / local | Chroma, LanceDB | Prototypes, notebooks, small apps |
| Managed cloud | Pinecone, Weaviate Cloud, Qdrant Cloud, Zilliz | Scale without ops |
| Self-hosted engines | Qdrant, Milvus, Weaviate | Large scale, control |
| Extensions to existing DBs | pgvector (Postgres), Elasticsearch/OpenSearch, MongoDB Atlas, Redis | Keep vectors next to your existing data |
| Libraries (not DBs) | FAISS, hnswlib, ScaNN | Raw speed, you handle storage |

**Choosing:** if you already run Postgres and have < tens of millions of vectors, pgvector with an HNSW index is often the simplest choice. Reach for a dedicated engine when you need very high scale, advanced hybrid search, or multi-tenancy at volume.

## Example
**Analogy:** A library catalog organized not alphabetically but by "what the book is about." Ask for a book like *this one* and the librarian walks you straight to the right shelf, and you can also say "only books published after 2020" (metadata filter).

```python
import chromadb

client = chromadb.PersistentClient(path="./vdb")       # persists to disk
col = client.get_or_create_collection(
    name="faq", metadata={"hnsw:space": "cosine"})     # default embedder: all-MiniLM-L6-v2

col.add(
    ids=["1", "2", "3"],
    documents=["Refunds take 5 business days.",
               "Shipping is free over $50.",
               "Reembolsos tardan 5 días hábiles."],
    metadatas=[{"lang": "en"}, {"lang": "en"}, {"lang": "es"}],
)

res = col.query(query_texts=["How long until I get my money back?"],
                n_results=1, where={"lang": "en"})
print(res["documents"])   # [['Refunds take 5 business days.']]
print(res["distances"])   # e.g. [[0.4x]]  (cosine distance, lower = closer)
```

## Common Mistakes / Misconceptions
- **"I need a specialized vector DB from day one."** For many apps, pgvector or Chroma is plenty.
- **Treating FAISS as a database.** It's an index library; persistence, metadata, and updates are on you.
- **Storing vectors without the source text and metadata**, making results unexplainable and unfilterable.
- **Ignoring re-indexing cost** when changing embedding models.

## Key Takeaways
- A vector DB = ANN index + storage + metadata + database operations.
- HNSW is the dominant index type; metadata filtering and hybrid search are standard features.
- Options range from embedded (Chroma) to managed (Pinecone) to extensions (pgvector).
- Pick based on scale, existing stack, and filtering/hybrid needs.

## Related Questions
- [How is a vector database different from a relational database?](../Week-07-Embeddings-and-Vector-Databases/11-How-is-a-vector-database-different-from-a-relational-database.md)
- [What is ChromaDB?](../Week-07-Embeddings-and-Vector-Databases/12-What-is-ChromaDB.md)
- [What is FAISS?](../Week-07-Embeddings-and-Vector-Databases/13-What-is-FAISS.md)
- [What is metadata filtering?](../Week-08-Retrieval-Augmented-Generation-RAG/13-What-is-metadata-filtering.md)
