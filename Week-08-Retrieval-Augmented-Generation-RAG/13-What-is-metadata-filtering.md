# What is metadata filtering?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 13

## Short Answer
Metadata filtering restricts a vector search to chunks whose attached metadata matches conditions, such as `department = "HR"`, `year >= 2026`, `language = "en"`, or `tenant_id = 42`, before or while ranking by similarity. It combines the precision of structured, database-style filters with the flexibility of semantic search. In RAG it's essential for relevance (right product/version), freshness (latest docs), and security (only documents the user may see).

## Detailed Explanation
**Metadata** = structured fields stored with each chunk at ingestion: source, title, section, page, author, date, version, language, document type, tenant, access groups.

**Common uses:**

| Goal | Filter example |
|---|---|
| Access control / multi-tenancy | `{"tenant_id": "acme"}`, `{"group": {"$in": user_groups}}` |
| Freshness | `{"year": {"$gte": 2026}}` |
| Product/version scoping | `{"product": "api", "version": "v3"}` |
| Language | `{"lang": "de"}` |
| Document type | `{"type": {"$in": ["policy", "faq"]}}` |

**Pre-filter vs post-filter:**
```
Pre-filter / filtered search:  apply filter → similarity search within matches → top-k   ✔ always k results (if they exist)
Post-filter:                   similarity top-k → drop non-matching                      ✘ may return 0–k results
```
Modern vector DBs (Chroma, Qdrant, Weaviate, Pinecone, Milvus, pgvector with `WHERE`) apply filters *during* ANN traversal. Very selective filters can lower HNSW recall; engines handle this with filter-aware search or fall back to brute force on the small matching subset.

**Where filters come from:**
- The app context (logged-in user's tenant and permissions): never trust the LLM for this.
- UI selections (product dropdown, date range).
- **Self-querying**: an LLM extracts filters from the question ("HR policies from 2026 about remote work" → `{"dept": "HR", "year": 2026}` + query "remote work").

**Chroma operators:** `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, combined with `$and` / `$or`; `where_document={"$contains": "..."}` filters on text content.

## Example
**Analogy:** Shopping online for "comfortable running shoes": the search understands what "comfortable" means (semantic), but you also tick "size 42," "under $100," "in stock" (metadata filters). Without the filters you'd see perfect shoes you can't buy.

```python
import chromadb

col = chromadb.Client().get_or_create_collection("policies", metadata={"hnsw:space": "cosine"})
col.add(
    ids=["a", "b", "c", "d"],
    documents=["Remote work allowed 2 days per week.",
               "Remote work allowed 3 days per week.",
               "Remote work allowed 3 days per week for Globex staff.",
               "Expense reports due within 30 days."],
    metadatas=[{"tenant": "acme", "year": 2024, "dept": "HR"},
               {"tenant": "acme", "year": 2026, "dept": "HR"},
               {"tenant": "globex", "year": 2026, "dept": "HR"},
               {"tenant": "acme", "year": 2026, "dept": "Finance"}],
)

user_tenant = "acme"   # from the authenticated session, NOT from the prompt
res = col.query(
    query_texts=["How many remote days can I take?"],
    n_results=1,
    where={"$and": [{"tenant": user_tenant}, {"year": {"$gte": 2026}}, {"dept": "HR"}]},
)
print(res["documents"])   # [['Remote work allowed 3 days per week.']]
print(res["ids"])         # [['b']]   -> old 2024 policy and other tenant excluded
```

## Common Mistakes / Misconceptions
- **Letting the LLM decide security filters.** Permissions must come from trusted application code.
- **Post-filtering a small top-k**, which leaves empty or tiny result sets.
- **Inconsistent metadata types** (e.g. `"2026"` string vs `2026` int), so range filters silently fail.
- **Forgetting metadata at ingestion**: you can't filter on fields you never stored.

## Key Takeaways
- Metadata filtering = structured conditions + semantic similarity in one query.
- Critical for access control, freshness, and scoping to the right product/version.
- Prefer filtered (pre-filter) search over post-filtering.
- Design metadata at ingestion time with consistent types.

## Related Questions
- [What is document ingestion?](../Week-08-Retrieval-Augmented-Generation-RAG/04-What-is-document-ingestion.md)
- [How does vector similarity retrieval work?](../Week-08-Retrieval-Augmented-Generation-RAG/09-How-does-vector-similarity-retrieval-work.md)
- [What is ChromaDB?](../Week-07-Embeddings-and-Vector-Databases/12-What-is-ChromaDB.md)
- [How is a vector database different from a relational database?](../Week-07-Embeddings-and-Vector-Databases/11-How-is-a-vector-database-different-from-a-relational-database.md)
