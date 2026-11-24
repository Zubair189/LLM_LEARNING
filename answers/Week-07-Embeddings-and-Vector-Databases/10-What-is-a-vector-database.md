# What is a vector database?

> Week 07 – Embeddings and Vector Databases · Question 10

## In One Line
A vector database is a database built to **store embeddings and quickly find the most similar ones**.

## Simple Explanation
Once you have thousands or millions of embeddings, you need somewhere to keep them and a fast way to search them. That is a vector database.

Each record usually holds four things:

1. **ID:** a unique name, like `"faq-042"`.
2. **Vector:** the embedding (e.g. 384 numbers).
3. **Document:** the original text, so you can show it.
4. **Metadata:** extra labels, like `{"source": "faq.pdf", "lang": "en"}`.

**What it does for you:**
- **Fast similarity search** using a special index (usually HNSW) instead of checking every vector.
- **Metadata filtering**, e.g. "only English documents".
- **Normal database jobs:** add, update and delete records, and save them to disk.

**Popular options:** Chroma (easy, great for learning), Pinecone (hosted in the cloud), Qdrant, Weaviate, Milvus, and **pgvector** (adds vector search to PostgreSQL). FAISS is a fast search **library**, not a full database.

**Link to LLMs:** in a RAG app, the vector database is the "memory" that holds your documents, ready to be searched for every question.

## Real-Life Example
A library where books are shelved by **what they are about**, not by title. You say "find me books like this one" and the librarian walks straight to the right shelf. You can also add "only books after 2020" (a metadata filter).

## Code Example
Using Chroma (needs `pip install chromadb`; not run here):

```python
import chromadb

client = chromadb.PersistentClient(path="./vdb")        # saves data to disk
col = client.get_or_create_collection(name="faq")        # like a table

col.add(
    ids=["1", "2", "3"],                                 # unique IDs
    documents=["Refunds take 5 business days.",          # the text
               "Shipping is free over $50.",
               "Reembolsos tardan 5 días hábiles."],
    metadatas=[{"lang": "en"}, {"lang": "en"}, {"lang": "es"}],  # labels
)                                                        # Chroma embeds the text for us

res = col.query(query_texts=["How long until I get my money back?"],
                n_results=1, where={"lang": "en"})       # search + filter
print(res["documents"])   # e.g. [['Refunds take 5 business days.']]
```

**What happened:** Chroma turned each text into an embedding, stored it with its labels, and then found the English document closest in meaning to the question.

## Watch Out
- You don't need a big, special database on day one. Chroma or pgvector is enough for many apps.
- Always store the **original text and metadata** next to the vector, or you can't show, cite or filter results.

## Remember
- Vector database = stored embeddings + fast similarity search.
- Records hold ID, vector, text and metadata.
- It is the knowledge store behind most RAG apps.

## Related Questions
- [How is a vector database different from a relational database?](../Week-07-Embeddings-and-Vector-Databases/11-How-is-a-vector-database-different-from-a-relational-database.md)
- [What is ChromaDB?](../Week-07-Embeddings-and-Vector-Databases/12-What-is-ChromaDB.md)
- [What is FAISS?](../Week-07-Embeddings-and-Vector-Databases/13-What-is-FAISS.md)
- [What is metadata filtering?](../Week-08-Retrieval-Augmented-Generation-RAG/13-What-is-metadata-filtering.md)
