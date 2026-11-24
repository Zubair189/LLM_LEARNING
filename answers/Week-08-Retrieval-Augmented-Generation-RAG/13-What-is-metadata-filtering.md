# What is metadata filtering?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 13

## In One Line
Metadata filtering means **only searching chunks whose labels match certain rules**, like "only HR documents from 2026".

## Simple Explanation
**Metadata** is extra information stored with each chunk: source file, department, date, language, product, or which customer it belongs to. It is attached during ingestion.

Normal vector search looks only at meaning. Metadata filtering adds **exact rules** on top, for example:

- `{"dept": "HR"}`: only HR documents.
- `{"year": {"$gte": 2026}}`: only 2026 or newer (`$gte` means "greater than or equal").
- `{"tenant": "acme"}`: only documents belonging to the customer "acme".

**Why it matters in RAG:**

1. **Right topic:** only search the product or version the user is asking about.
2. **Fresh info:** skip old versions of a policy.
3. **Security:** a user must only see documents they are allowed to see.

**Filter during the search, not after.** If you first take the top 3 and then filter, you might be left with nothing. Vector databases like Chroma apply the filter **while** searching, so you still get up to 3 good results.

**Important:** security filters (like which customer the user belongs to) must come from your app's login system, **never** from the LLM or the user's question.

**Link to LLMs:** filtering makes sure the LLM only sees the right, allowed, up-to-date chunks.

## Real-Life Example
Online shopping for "comfortable running shoes". The search understands "comfortable" (meaning), but you also tick "size 42", "under $100" and "in stock" (filters). Without the filters you'd see perfect shoes you can't buy.

## Code Example
Needs `pip install chromadb` (not run here, so the output is an example):

```python
import chromadb

col = chromadb.Client().get_or_create_collection("policies")
col.add(
    ids=["a", "b", "c"],
    documents=["Remote work allowed 2 days per week.",
               "Remote work allowed 3 days per week.",
               "Remote work allowed 4 days per week."],
    metadatas=[{"tenant": "acme", "year": 2024},     # old policy
               {"tenant": "acme", "year": 2026},     # current policy
               {"tenant": "globex", "year": 2026}],  # another company
)

user_tenant = "acme"   # comes from the login system, NOT from the prompt
res = col.query(
    query_texts=["How many remote days can I take?"],
    n_results=1,
    where={"$and": [{"tenant": user_tenant}, {"year": {"$gte": 2026}}]},
)
print(res["ids"])   # e.g. [['b']]  -> old policy and other company excluded
```

**What happened:** all three chunks are about remote work, but the filter kept only acme's 2026 policy.

## Watch Out
- Never let the LLM decide security filters. They must come from trusted app code.
- Store metadata with consistent types. If the year is sometimes `"2026"` (text) and sometimes `2026` (number), filters may silently fail.

## Remember
- Metadata = labels stored with each chunk.
- Filtering = exact rules + meaning search together.
- Use it for topic, freshness and security.

## Related Questions
- [What is document ingestion?](../Week-08-Retrieval-Augmented-Generation-RAG/04-What-is-document-ingestion.md)
- [How does vector similarity retrieval work?](../Week-08-Retrieval-Augmented-Generation-RAG/09-How-does-vector-similarity-retrieval-work.md)
- [What is ChromaDB?](../Week-07-Embeddings-and-Vector-Databases/12-What-is-ChromaDB.md)
- [How is a vector database different from a relational database?](../Week-07-Embeddings-and-Vector-Databases/11-How-is-a-vector-database-different-from-a-relational-database.md)
