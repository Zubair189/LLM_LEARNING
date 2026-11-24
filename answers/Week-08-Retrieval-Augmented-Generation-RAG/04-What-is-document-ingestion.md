# What is document ingestion?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 4

## In One Line
Document ingestion is the **preparation step** that turns raw files into clean, searchable chunks stored in your vector database.

## Simple Explanation
Before a RAG system can answer anything, your documents must be loaded and stored in a searchable form. That setup work is called **ingestion**. It is usually done once, then repeated when documents change.

**The steps:**

1. **Load:** read the files (PDF, Word, web pages, wiki pages…).
2. **Extract:** pull out the plain text, including text in tables.
3. **Clean:** remove junk like page numbers, menus and repeated headers.
4. **Chunk:** split the text into small pieces (about a paragraph).
5. **Add metadata:** attach labels to each chunk, like source file, page and date.
6. **Embed:** turn each chunk into an embedding (a list of numbers).
7. **Store:** save the ID, embedding, text and metadata in the vector database.

```
files → load → clean → chunk → add labels → embed → store
```

**Why it matters:** retrieval can only find what ingestion stored well. Messy text or missing chunks mean bad answers later, no matter how good the LLM is.

**Two good habits:**
- Give each chunk a **stable ID**, like `"hr.txt#0"`, so re-running ingestion updates chunks instead of creating duplicates.
- Keep **metadata** so you can cite sources and filter results later.

**Link to LLMs:** ingestion is the "write" side of RAG. Retrieval and the LLM are the "read" side.

## Real-Life Example
A library receiving boxes of donated books. Staff unpack them (load), remove torn covers (clean), record title and subject for each (metadata), and put them on the right shelf (store). If they skip steps, the books are effectively lost.

## Code Example
```python
import re

raw_docs = {   # 1. LOAD: pretend we read these files
    "hr.txt": "Vacation: 25 days per year.\n\nSick leave: 10 days.\n\nPage 1 of 1",
    "it.txt": "Reset passwords at id.example.com.\n\nVPN is required off-site.",
}

store = []
for name, text in raw_docs.items():
    text = re.sub(r"Page \d+ of \d+", "", text)                    # 2. CLEAN: drop footer
    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]  # 3. CHUNK by paragraph
    for n, chunk in enumerate(chunks):
        store.append({"id": f"{name}#{n}",                         # 4. stable ID
                      "text": chunk,
                      "source": name})                             #    + metadata
        # 5. EMBED + STORE: a real system would embed `chunk` and save it in a vector DB

print(len(store))        # 4
print(store[0])
# {'id': 'hr.txt#0', 'text': 'Vacation: 25 days per year.', 'source': 'hr.txt'}
```

**What happened:** two files became 4 clean chunks, each with an ID and a source label. The "Page 1 of 1" footer was removed.

## Watch Out
- Ingestion is not a one-time script. Documents change, so you need to update and delete chunks too.
- Always **look at** the extracted text. PDFs with tables or columns often come out garbled.

## Remember
- Ingestion = load → clean → chunk → label → embed → store.
- Its quality sets the limit for the whole RAG system.
- Use stable IDs and keep metadata.

## Related Questions
- [What is document chunking?](../Week-08-Retrieval-Augmented-Generation-RAG/05-What-is-document-chunking.md)
- [What is metadata filtering?](../Week-08-Retrieval-Augmented-Generation-RAG/13-What-is-metadata-filtering.md)
- [What is a RAG pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/11-What-is-a-RAG-pipeline.md)
- [What is ChromaDB?](../Week-07-Embeddings-and-Vector-Databases/12-What-is-ChromaDB.md)
