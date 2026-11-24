# What is document ingestion?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 4

## Short Answer
Document ingestion is the offline (or scheduled) process that turns raw source files into searchable chunks in your vector store: load documents (PDF, HTML, Word, Confluence, databases), extract and clean text, chunk it, attach metadata, embed each chunk, and upsert everything into the index. It's the "write path" of RAG, and its quality caps everything downstream: retrieval can't find what ingestion mangled or dropped.

## Detailed Explanation
```
 sources ─► LOAD ─► PARSE/EXTRACT ─► CLEAN ─► CHUNK ─► ENRICH METADATA ─► EMBED ─► UPSERT
 (S3, wiki,  (connectors) (PDF text,   (strip    (size,   (source, page,     (batch)  (vector DB,
  Drive, DB)              tables, OCR) headers,  overlap)  date, ACL, hash)            dedupe by id)
                                       dedupe)
```

| Stage | What happens | Pitfalls |
|---|---|---|
| Load | Connectors pull files/pages | Missed sources, auth issues |
| Parse | Extract text, tables, headings; OCR scans | Garbled tables, multi-column PDFs |
| Clean | Remove boilerplate, nav menus, repeated headers/footers | Over-cleaning removes content |
| Chunk | Split into retrievable units | Too big/small, broken context |
| Metadata | source, title, section, page, date, permissions | Missing ACLs leak data |
| Embed | Batch encode with embedding model | Rate limits, model mismatch |
| Upsert | Store id, vector, text, metadata | Duplicates, stale versions |

**Production concerns:**
- **Incremental updates:** compute a content hash per document; only re-embed changed files; delete chunks of removed docs.
- **Deterministic IDs:** e.g. `f"{doc_id}#chunk{n}"`, so re-runs `upsert` instead of duplicating.
- **Parsing tools:** `pypdf`, `unstructured`, `docling`, LlamaParse, cloud document AI; multimodal LLMs increasingly parse complex PDFs, tables, and images.
- **Contextual enrichment:** prepend a document title/section summary to each chunk ("contextual retrieval") to improve embedding quality.
- **Security:** carry access-control metadata from the source so retrieval can filter by user.

## Example
**Analogy:** A library receiving donated books. Before anyone can find them, staff unpack boxes (load), remove torn covers (clean), catalog each book with title, author, and subject (metadata), and shelve them in the right place (index). A sloppy intake process means books are effectively lost.

```python
import hashlib, re
import chromadb

raw_docs = {
    "hr-policy.txt": "HR POLICY v3\n\nVacation: 25 days per year.\n\nSick leave: 10 days.\n\nPage 1 of 1",
    "it-faq.txt":    "IT FAQ\n\nReset passwords at id.example.com.\n\nVPN is required off-site.",
}

def clean(text):
    text = re.sub(r"Page \d+ of \d+", "", text)       # strip footer boilerplate
    return text.strip()

def chunk(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]   # paragraph chunks

col = chromadb.PersistentClient(path="./kb").get_or_create_collection("kb")
for name, text in raw_docs.items():
    doc_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
    for n, c in enumerate(chunk(clean(text))):
        col.upsert(ids=[f"{name}#{n}"], documents=[c],
                   metadatas=[{"source": name, "chunk": n, "hash": doc_hash}])

print(col.count())   # 6  (HR: title, vacation, sick leave; IT: title, reset, VPN; footer removed)
```

Notice the tiny title-only chunks like "IT FAQ": a classic ingestion smell. Better to merge headings into the following chunk as context, so each chunk is self-explanatory.

## Common Mistakes / Misconceptions
- **Treating ingestion as a one-time script.** Sources change; you need incremental sync and deletions.
- **Ignoring parsing quality.** Garbage text in → garbage retrieval out; inspect extracted text manually.
- **Dropping metadata** (source, page, date, permissions), which blocks citations and filtering.
- **Random IDs** that create duplicates on every re-run.

## Key Takeaways
- Ingestion = load → parse → clean → chunk → enrich → embed → store.
- It's the foundation of RAG quality; inspect its output.
- Use deterministic IDs, content hashes, and incremental updates.
- Preserve metadata for citations, filtering, and access control.

## Related Questions
- [What is document chunking?](../Week-08-Retrieval-Augmented-Generation-RAG/05-What-is-document-chunking.md)
- [What is metadata filtering?](../Week-08-Retrieval-Augmented-Generation-RAG/13-What-is-metadata-filtering.md)
- [What is a RAG pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/11-What-is-a-RAG-pipeline.md)
- [What is ChromaDB?](../Week-07-Embeddings-and-Vector-Databases/12-What-is-ChromaDB.md)
