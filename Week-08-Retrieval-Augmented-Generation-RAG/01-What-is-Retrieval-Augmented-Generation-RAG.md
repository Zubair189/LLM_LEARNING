# What is Retrieval-Augmented Generation (RAG)?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 1

## Short Answer
RAG is a pattern where, before an LLM answers, the system *retrieves* relevant passages from an external knowledge source (documents, a vector database, a search engine) and *injects* them into the prompt, so the model generates an answer grounded in that evidence. It lets an LLM use private, up-to-date, or domain-specific information without retraining, reduces hallucinations, and allows citing sources. The term comes from Lewis et al. (2020); today it's the most common architecture for enterprise LLM apps.

## Detailed Explanation
```
             ┌──────────── Retrieval ────────────┐   ┌──── Augmentation ────┐   ┌─ Generation ─┐
user query ─►│ embed query → search index → top-k│──►│ build prompt with    │──►│ LLM writes   │─► answer
             │ (hybrid + rerank + filters)       │   │ chunks + instructions│   │ grounded     │   + citations
             └───────────────────────────────────┘   └──────────────────────┘   └──────────────┘
                         ▲
        offline: ingest → chunk → embed → store in vector DB
```

**Three parts of the name:**
- **Retrieval**: find the most relevant chunks (vector search, often hybrid with BM25, then reranking).
- **Augmented**: add those chunks to the prompt as context ("context injection").
- **Generation**: the LLM composes an answer using that context, ideally citing it.

**Why it's popular:**

| Benefit | Why |
|---|---|
| Fresh knowledge | Update the index, not the model |
| Private data | Company docs never enter model weights |
| Fewer hallucinations | Model answers from evidence |
| Traceability | Cite source chunks |
| Cost | Far cheaper than fine-tuning for knowledge |
| Access control | Filter retrieval by user permissions |

**Variants in 2026:** naive RAG (single retrieve-then-generate), advanced RAG (query rewriting, hybrid search, reranking), **agentic RAG** (an agent decides when/what to search, iterates, uses multiple tools), and GraphRAG (knowledge-graph-aware retrieval). Long-context models (hundreds of thousands to ~1M tokens) haven't killed RAG: retrieval is still cheaper, faster, more precise, and permission-aware at scale.

## Example
**Analogy:** An open-book exam. The student (LLM) is smart but doesn't memorize your company handbook. RAG hands them the right pages before they answer, and asks them to cite the page.

```python
# pip install chromadb anthropic
import chromadb, anthropic

col = chromadb.Client().get_or_create_collection("handbook")
col.add(ids=["1", "2", "3"], documents=[
    "Employees receive 25 days of paid vacation per year.",
    "Unused vacation days up to 5 can be carried over to next year.",
    "The office is closed on December 25 and January 1.",
])

question = "How many vacation days can I carry over?"
hits = col.query(query_texts=[question], n_results=2)["documents"][0]   # Retrieval
context = "\n".join(f"[{i+1}] {h}" for i, h in enumerate(hits))
prompt = f"Context:\n{context}\n\nQuestion: {question}"                  # Augmentation

resp = anthropic.Anthropic().messages.create(                            # Generation
    model="claude-sonnet-5-5", max_tokens=1024,
    system="Answer only from the provided context. Cite sources like [1]. "
           "If the answer isn't in the context, say you don't know.",
    messages=[{"role": "user", "content": prompt}],
)
print(resp.content[0].text)
# e.g. "You can carry over up to 5 unused vacation days to the next year [1]."
```

## Common Mistakes / Misconceptions
- **"RAG eliminates hallucinations."** It reduces them; bad retrieval or ignored context still produces wrong answers.
- **"RAG teaches the model."** Nothing is learned; knowledge is supplied per request.
- **Blaming the LLM first.** Most RAG failures are retrieval failures (bad chunking, wrong k, no hybrid search).
- **Skipping evaluation.** Measure retrieval (context precision/recall) and generation (faithfulness, answer relevance).

## Key Takeaways
- RAG = retrieve relevant context → inject into prompt → generate grounded answer.
- It provides fresh, private, citable knowledge without retraining.
- Retrieval quality is the biggest lever on answer quality.
- Modern RAG adds hybrid search, reranking, agentic loops, and systematic evaluation.

## Related Questions
- [Why do LLMs need external knowledge?](../Week-08-Retrieval-Augmented-Generation-RAG/02-Why-do-LLMs-need-external-knowledge.md)
- [How is RAG different from fine-tuning?](../Week-08-Retrieval-Augmented-Generation-RAG/03-How-is-RAG-different-from-fine-tuning.md)
- [What is a RAG pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/11-What-is-a-RAG-pipeline.md)
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
