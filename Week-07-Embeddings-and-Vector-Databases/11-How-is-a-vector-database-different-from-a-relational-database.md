# How is a vector database different from a relational database?

> Week 07 – Embeddings and Vector Databases · Question 11

## Short Answer
A relational database (PostgreSQL, MySQL) stores structured rows and answers *exact* questions with SQL: "orders where status = 'shipped'." A vector database stores embeddings and answers *similarity* questions: "the 5 passages most similar in meaning to this query," returning ranked, approximate results. Relational DBs use B-tree/hash indexes for exact lookups; vector DBs use ANN indexes like HNSW. The line is blurring: pgvector adds vector search to Postgres, and most vector DBs support metadata filters.

## Detailed Explanation
| Aspect | Relational DB | Vector DB |
|---|---|---|
| Data model | Tables, rows, typed columns, foreign keys | Vectors + payload/metadata (+ text) |
| Query type | Exact match, ranges, joins, aggregates | k-nearest-neighbour by similarity |
| Query language | SQL | API calls (`query`, `search`), sometimes SQL-like |
| Result | Exact set (all matching rows) | Ranked top-k with scores |
| Correctness | Deterministic, exact | Approximate (ANN recall < 100%) |
| Index | B-tree, hash, GIN | HNSW, IVF, PQ, DiskANN |
| Transactions | Strong ACID | Varies; often eventual consistency |
| Strength | Business records, reporting, integrity | Semantic search, RAG, recommendations |

**Why B-trees don't help:** a B-tree sorts values along one dimension. "Nearest" in 384 dimensions has no single sort order, so similarity search needs graph or cluster-based indexes.

**Same question, two worlds:**
```sql
-- Relational: exact
SELECT title FROM articles WHERE category = 'billing' AND year >= 2025;

-- pgvector (Postgres extension): similarity + filter in one query
SELECT title FROM articles
WHERE year >= 2025
ORDER BY embedding <=> '[0.02, -0.11, ...]'::vector   -- cosine distance
LIMIT 5;
```

**In practice they work together.** Apps keep users, orders, and permissions in a relational DB, and chunks + embeddings in a vector store (or in the same Postgres via pgvector). The vector result IDs are joined back to relational data for display and access control.

## Example
**Analogy:** A relational DB is a filing cabinet with labelled folders: you find "Invoice #1042" instantly, but can't ask for "invoices that *feel like* this one." A vector DB is a librarian who understands topics: great at "more like this," but it won't guarantee every match.

```python
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

docs = ["Refund takes 5 days", "Free shipping over $50", "Money back guarantee"]

# Relational: exact keyword match
db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE faq(id INTEGER, text TEXT)")
db.executemany("INSERT INTO faq VALUES (?, ?)", list(enumerate(docs)))
print(db.execute("SELECT text FROM faq WHERE text LIKE '%reimburse%'").fetchall())
# []  -> no exact word match

# Vector: similarity search
m = SentenceTransformer("all-MiniLM-L6-v2")
E = m.encode(docs, normalize_embeddings=True)
q = m.encode("Can I get reimbursed?", normalize_embeddings=True)
print([docs[i] for i in np.argsort(-(E @ q))[:2]])
# likely ['Money back guarantee', 'Refund takes 5 days']
```

## Common Mistakes / Misconceptions
- **"Vector DBs replace relational DBs."** They complement them; you still need transactions and exact queries.
- **Expecting exact, complete results** from ANN search. It returns top-k approximate neighbours.
- **Ignoring pgvector.** If you already run Postgres, adding vectors there is often simplest.
- **Not enforcing permissions** in the vector layer; filter by tenant/user metadata.

## Key Takeaways
- Relational = exact, structured, SQL; vector = similarity, ranked, approximate.
- Different index structures: B-tree vs HNSW/IVF.
- Real systems combine both (or use pgvector for both in one place).
- Metadata filters bring some relational-style filtering into vector search.

## Related Questions
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
- [What is approximate nearest neighbor search?](../Week-07-Embeddings-and-Vector-Databases/14-What-is-approximate-nearest-neighbor-search.md)
- [What is metadata filtering?](../Week-08-Retrieval-Augmented-Generation-RAG/13-What-is-metadata-filtering.md)
