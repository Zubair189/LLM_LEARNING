# What is ChromaDB?

> Week 07 – Embeddings and Vector Databases · Question 12

## Short Answer
ChromaDB (Chroma) is an open-source, developer-friendly vector database designed for LLM apps. It runs in-process in Python (in-memory or persisted to disk), as a client-server, or as a hosted cloud service, and it can embed your documents automatically (default: `all-MiniLM-L6-v2`). You add documents with IDs and metadata to a *collection*, then query by text with optional metadata filters, which makes it a popular choice for prototyping RAG.

## Detailed Explanation
**Core concepts:**

| Concept | Meaning |
|---|---|
| Client | `chromadb.Client()` (in-memory), `PersistentClient(path=...)` (local disk), `HttpClient(host=...)` (server), `CloudClient` (Chroma Cloud) |
| Collection | Like a table: holds ids, embeddings, documents, metadatas |
| Embedding function | Converts text → vectors; default is MiniLM via ONNX, or plug in OpenAI, sentence-transformers, etc. |
| Distance space | Set via collection metadata `{"hnsw:space": "cosine"}`; default is `"l2"` (newer versions also accept a `configuration` dict) |
| Index | HNSW under the hood |

**Main operations:**
- `add` / `upsert` / `update` / `delete`: write data (ids must be unique strings).
- `query(query_texts=[...], n_results=k, where={...}, where_document={...})`: similarity search with metadata filters (`$eq`, `$gte`, `$in`, `$and`, `$or`) and document-content filters (`$contains`).
- `get(where=...)`: fetch by id or filter without similarity.

**Query result shape:** a dict of lists-of-lists (one inner list per query): `ids`, `documents`, `metadatas`, `distances`.

**Strengths:** minimal setup, Pythonic API, auto-embedding, metadata filtering, integrations with LangChain/LlamaIndex. **Limitations:** for very large scale or demanding multi-tenant production, teams often move to Chroma's distributed/cloud offering, Qdrant, Milvus, Pinecone, or pgvector. Chroma has also been adding full-text/sparse search to support hybrid retrieval, so check your version's docs.

## Example
**Analogy:** Chroma is like SQLite for embeddings: a lightweight store you can drop into a project in two lines, good enough for a lot of real apps.

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_store")
col = client.get_or_create_collection(
    name="policies", metadata={"hnsw:space": "cosine"})

col.upsert(
    ids=["p1", "p2", "p3", "p4"],
    documents=[
        "Employees get 25 days of paid leave per year.",
        "Remote work is allowed up to 3 days a week.",
        "Expense claims must be filed within 30 days.",
        "Parental leave is 16 weeks at full pay.",
    ],
    metadatas=[{"dept": "HR", "year": 2026}, {"dept": "HR", "year": 2025},
               {"dept": "Finance", "year": 2026}, {"dept": "HR", "year": 2026}],
)

res = col.query(
    query_texts=["How much vacation do I get?"],
    n_results=2,
    where={"$and": [{"dept": "HR"}, {"year": {"$gte": 2026}}]},
)
for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(f"{dist:.2f}  {doc}")
# approx:
# 0.4x  Employees get 25 days of paid leave per year.
# 0.7x  Parental leave is 16 weeks at full pay.
print(col.count())   # 4
```

## Common Mistakes / Misconceptions
- **Forgetting persistence.** `chromadb.Client()` is in-memory; data vanishes on restart. Use `PersistentClient`.
- **Leaving the default L2 space** when you meant cosine; set `hnsw:space` at creation (it can't be changed later).
- **Changing embedding functions** on an existing collection: old and new vectors become incompatible.
- **Using `add` for re-ingestion**: duplicate IDs are an error or ignored; use `upsert`.

## Key Takeaways
- Chroma is an open-source, Python-first vector DB, great for prototypes and small-to-medium apps.
- Collections hold ids, documents, embeddings, and metadata.
- `query` supports top-k similarity plus `where` metadata filters.
- Configure persistence and distance metric deliberately.

## Related Questions
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
- [What is FAISS?](../Week-07-Embeddings-and-Vector-Databases/13-What-is-FAISS.md)
- [How do you retrieve similar documents using embeddings?](../Week-07-Embeddings-and-Vector-Databases/15-How-do-you-retrieve-similar-documents-using-embeddings.md)
- [What is metadata filtering?](../Week-08-Retrieval-Augmented-Generation-RAG/13-What-is-metadata-filtering.md)
