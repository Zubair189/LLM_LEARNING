# What is a RAG pipeline?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 11

## Short Answer
A RAG pipeline is the complete end-to-end system that makes Retrieval-Augmented Generation work: an **offline indexing** path (ingest → chunk → embed → store) and an **online query** path (retrieve → rerank → inject context → generate → return answer with citations), plus evaluation and monitoring around it. It connects your data, a vector store, and an LLM into one flow that turns a question into a grounded answer.

## Detailed Explanation
**Scope note:** the *retrieval pipeline* (Q8) is just the part that returns ranked chunks. The *RAG pipeline* includes it, plus ingestion before it and context injection, generation, citation, guardrails, and evaluation after it.

```
OFFLINE / INDEXING                                ONLINE / QUERY
──────────────────                                ──────────────
sources                                           user question
  │ load + parse                                    │ query rewrite (optional)
  ▼                                                 ▼
clean ─► chunk (+overlap) ─► metadata         ┌── retrieval pipeline ──────────────┐
  │                                           │ hybrid search (vector + BM25)      │
  ▼                                           │ metadata/permission filters        │
embed ─► vector DB  ◄────────────────────────►│ rerank → top 3–8 chunks            │
                                              └────────────────────────────────────┘
                                                    ▼
                                              context injection (prompt build)
                                                    ▼
                                              LLM generation (grounded, cited)
                                                    ▼
                                              guardrails / citation check ─► answer
                                                    ▼
                                              logging + evaluation (faithfulness, relevance)
```

| Component | Common choices (2026) |
|---|---|
| Parsing | unstructured, docling, LlamaParse, multimodal LLMs |
| Embeddings | all-MiniLM-L6-v2 (learning), BGE/E5/GTE, Voyage, OpenAI, Cohere |
| Store | Chroma, pgvector, Qdrant, Weaviate, Pinecone, Elasticsearch |
| Reranker | cross-encoder ms-marco, bge-reranker, Cohere/Voyage rerank |
| Generator | Claude, GPT, Gemini, open-weight models |
| Frameworks | LlamaIndex, LangChain/LangGraph, Haystack, or plain Python |
| Evaluation | Ragas, DeepEval, TruLens, LLM-as-judge, golden test sets |

**Evolution:** *naive RAG* (retrieve once, generate) → *advanced RAG* (query rewriting, hybrid, reranking) → *agentic RAG* (an agent decides when to search, which source/tool to use, and iterates until it has enough evidence).

## Example
**Analogy:** A restaurant. Offline: buying, washing, and prepping ingredients into labelled containers (ingestion/indexing). Online: an order comes in, the cook grabs the right containers (retrieval), arranges them on the station (context injection), cooks (generation), and the expediter checks the plate before it goes out (guardrails/evaluation).

Minimal end-to-end RAG pipeline:

```python
import chromadb, anthropic

# ---------- OFFLINE: index ----------
docs = {"returns.md": "Items can be returned within 30 days. Refunds go to the original card.",
        "shipping.md": "Standard shipping takes 3-5 days. Express shipping takes 1 day."}
col = chromadb.PersistentClient(path="./rag_db").get_or_create_collection(
    "shop", metadata={"hnsw:space": "cosine"})
for src, text in docs.items():
    for n, sent in enumerate(s.strip() for s in text.split(". ") if s.strip()):
        col.upsert(ids=[f"{src}#{n}"], documents=[sent], metadatas=[{"source": src}])

# ---------- ONLINE: answer ----------
def answer(question, k=3):
    res = col.query(query_texts=[question], n_results=k)                 # retrieve
    ctx = "\n".join(f"[{i+1}] ({m['source']}) {d}"
                    for i, (d, m) in enumerate(zip(res["documents"][0], res["metadatas"][0])))
    resp = anthropic.Anthropic().messages.create(                        # generate
        model="claude-sonnet-5-5", max_tokens=1024,
        system="Answer only from the provided context. Cite like [1]. "
               "If the answer is not there, say you don't know.",
        messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}],
    )
    return resp.content[0].text

print(answer("How long do I have to return something?"))
# e.g. "You can return items within 30 days [1]."
```

## Common Mistakes / Misconceptions
- **Treating RAG as "vector DB + LLM."** Ingestion quality, reranking, prompting, and evaluation matter as much.
- **No evaluation loop.** Without a test set, you can't tell whether a change helped.
- **Rebuilding the index from scratch** on every update instead of incremental upserts.
- **Over-relying on frameworks** without understanding each stage; debugging becomes guesswork.

## Key Takeaways
- RAG pipeline = offline indexing + online retrieve → inject → generate, with evaluation around it.
- The retrieval pipeline is one stage inside it.
- Each stage is independently tunable and testable.
- Modern pipelines add hybrid search, reranking, guardrails, and agentic loops.

## Related Questions
- [What is a retrieval pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
- [What is document ingestion?](../Week-08-Retrieval-Augmented-Generation-RAG/04-What-is-document-ingestion.md)
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
- [What is Retrieval-Augmented Generation (RAG)?](../Week-08-Retrieval-Augmented-Generation-RAG/01-What-is-Retrieval-Augmented-Generation-RAG.md)
