# How is a vector database different from a relational database?

> Week 07 – Embeddings and Vector Databases · Question 11

## In One Line
A relational database finds rows that **exactly match** your conditions; a vector database finds items that are **most similar in meaning**.

## Simple Explanation
A **relational database** (like PostgreSQL or MySQL) stores data in tables with rows and columns. You ask questions in **SQL**, such as "give me all orders where status = 'shipped'". The answer is exact: a row matches or it doesn't.

A **vector database** stores embeddings. You ask "give me the 5 passages closest in meaning to this question". The answer is a **ranked list with scores**, from most to least similar.

Key differences in plain words:

1. **Type of question:** exact match vs "most similar".
2. **Result:** every matching row vs the top-k best matches.
3. **Accuracy:** relational is always exact; vector search is often **approximate** (it may very rarely miss the true best match, in exchange for speed).
4. **Best for:** business records (users, orders, payments) vs semantic search and RAG.

**They work together.** Real apps keep users and orders in a relational database and document chunks in a vector database. Some tools do both: **pgvector** adds vector search to PostgreSQL.

**Link to LLMs:** a RAG app uses the vector database to find relevant text, and often a relational database to check things like user permissions.

## Real-Life Example
A relational database is a filing cabinet with labelled folders: you find "Invoice #1042" instantly, but you can't ask for "invoices that feel like this one". A vector database is a librarian who understands topics: great at "more like this", but it gives the best guesses, not a complete exact list.

## Code Example
```python
import sqlite3

docs = ["Refund takes 5 days", "Free shipping over $50", "Money back guarantee"]

# Relational database: exact word matching with SQL
db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE faq(text TEXT)")
db.executemany("INSERT INTO faq VALUES (?)", [(d,) for d in docs])
print(db.execute("SELECT text FROM faq WHERE text LIKE '%reimburse%'").fetchall())
# []  <- no row contains the exact word "reimburse"

# Vector way: tiny made-up vectors [money_back, shipping]
vecs = [[0.9, 0.1], [0.1, 0.9], [0.95, 0.05]]
q = [1.0, 0.0]                                # "Can I get reimbursed?"
scores = [q[0] * v[0] + q[1] * v[1] for v in vecs]
best = sorted(zip(scores, docs), reverse=True)[:2]
print([d for s, d in best])
# ['Money back guarantee', 'Refund takes 5 days']
```

**What happened:** SQL found nothing because no text contains "reimburse". The vector search found the two documents that **mean** the same thing.

## Watch Out
- Vector databases do **not** replace relational ones. You still need exact queries and safe transactions for things like payments.
- Don't expect a complete list from vector search. It returns the top-k closest, not "all matches".

## Remember
- Relational = exact matches with SQL.
- Vector = similar meaning, ranked by score.
- Real systems usually use both.

## Related Questions
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
- [What is approximate nearest neighbor search?](../Week-07-Embeddings-and-Vector-Databases/14-What-is-approximate-nearest-neighbor-search.md)
- [What is metadata filtering?](../Week-08-Retrieval-Augmented-Generation-RAG/13-What-is-metadata-filtering.md)
