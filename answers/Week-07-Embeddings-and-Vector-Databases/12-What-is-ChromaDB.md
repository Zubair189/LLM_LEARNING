# What is ChromaDB?

> Week 07 – Embeddings and Vector Databases · Question 12

## In One Line
ChromaDB (Chroma) is a **free, easy-to-use vector database for Python**, popular for building and learning RAG.

## Simple Explanation
Chroma lets you store texts and search them by meaning with just a few lines of code. It is open source, so anyone can use it for free.

**The main ideas:**

1. **Client:** your connection to Chroma.
   - `chromadb.Client()` keeps data in memory only (lost when the program stops).
   - `chromadb.PersistentClient(path="...")` saves data to disk.
2. **Collection:** like a table. It holds IDs, texts, embeddings and metadata.
3. **Automatic embeddings:** if you give Chroma plain text, it turns it into embeddings for you (by default with the `all-MiniLM-L6-v2` model).
4. **Query:** you ask a question in text, and Chroma returns the closest matches. You can add a `where` filter on metadata, like "only HR documents".

**Why people like it:** almost no setup, simple Python API, and it works well with tools like LangChain and LlamaIndex. For very large production systems, teams sometimes move to bigger options like Qdrant, Pinecone or pgvector.

**Link to LLMs:** Chroma is often the "knowledge store" in a first RAG project. You put your documents in, and fetch the best chunks for each question.

## Real-Life Example
Chroma is like a notebook you can search by meaning. You jot down notes, each with a label. Later you ask, "What did I write about vacation?" and it opens to the right pages, even if you never wrote the word "vacation".

## Code Example
Needs `pip install chromadb` (not run here, so the output is an example):

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_store")   # save to disk
col = client.get_or_create_collection(name="policies")       # create or open

col.add(
    ids=["p1", "p2", "p3"],                                   # unique IDs
    documents=["Employees get 25 days of paid leave per year.",
               "Remote work is allowed up to 3 days a week.",
               "Expense claims must be filed within 30 days."],
    metadatas=[{"dept": "HR"}, {"dept": "HR"}, {"dept": "Finance"}],
)

res = col.query(
    query_texts=["How much vacation do I get?"],              # question in plain text
    n_results=1,                                              # top-1 result
    where={"dept": "HR"},                                     # only HR documents
)
print(res["documents"])   # e.g. [['Employees get 25 days of paid leave per year.']]
```

**What happened:** Chroma embedded the three policies, then found the HR policy closest in meaning to "vacation", even though the text says "paid leave".

## Watch Out
- `chromadb.Client()` is **in memory only**. Use `PersistentClient` if you want to keep your data.
- Don't change the embedding model on an existing collection. Old and new vectors won't match; create a new collection instead.

## Remember
- Chroma = simple, free, Python-first vector database.
- Collections store IDs, texts, embeddings and metadata.
- `query` finds similar texts; `where` filters by metadata.

## Related Questions
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
- [What is FAISS?](../Week-07-Embeddings-and-Vector-Databases/13-What-is-FAISS.md)
- [How do you retrieve similar documents using embeddings?](../Week-07-Embeddings-and-Vector-Databases/15-How-do-you-retrieve-similar-documents-using-embeddings.md)
- [What is metadata filtering?](../Week-08-Retrieval-Augmented-Generation-RAG/13-What-is-metadata-filtering.md)
