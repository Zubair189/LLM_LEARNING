# What is the difference between retrieval and generation?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 14

## Short Answer
Retrieval *finds* existing information: it searches an index and returns stored chunks that are relevant to the query, without creating any new text. Generation *creates* new text: an LLM reads the question plus retrieved context and writes an answer token by token. In RAG, retrieval decides *what evidence* the model sees; generation decides *how that evidence is turned into an answer*. They fail differently, use different models, and are evaluated with different metrics.

## Detailed Explanation
| Aspect | Retrieval | Generation |
|---|---|---|
| Purpose | Find relevant existing content | Produce a new, fluent answer |
| Output | Ranked list of chunks (verbatim) | Novel text |
| Typical models | Embedding model, BM25, reranker | LLM (Claude, GPT, Gemini, Llama...) |
| Deterministic? | Mostly (same query → same results) | Probabilistic (sampling) |
| Speed/cost | Milliseconds, cheap | Seconds, more expensive per token |
| Can it invent facts? | No, only returns what's stored | Yes, can hallucinate |
| Failure mode | Misses relevant chunk, returns noise | Ignores/distorts context, unsupported claims |
| Metrics | Recall@k, MRR, nDCG, **context precision**, **context recall** | **Faithfulness/groundedness**, **answer relevance**, correctness |
| Tuning levers | Chunking, embeddings, hybrid search, k, filters, reranking | Prompt, model choice, temperature, citation format |

```
question ─► RETRIEVAL ─► [chunk1, chunk2, chunk3] ─► GENERATION ─► "Answer ... [1][3]"
            "librarian"                                "writer"
```

**Why the distinction matters for debugging:** when an answer is wrong, first ask *"Was the right information in the retrieved context?"*
- **No** → retrieval problem (fix chunking, embeddings, hybrid search, k, filters).
- **Yes** → generation problem (fix prompt, model, citation requirements).

**Both are needed:** retrieval alone returns raw passages the user must read; generation alone relies on parametric memory and may hallucinate. RAG combines them: grounded *and* fluent. In agentic RAG, the generator can also decide to call retrieval again (multi-step search).

## Example
**Analogy:** A research assistant (retrieval) pulls the three most relevant articles from the archive. A journalist (generation) reads them and writes a clear summary for readers. A great writer with the wrong articles produces a confident but wrong story; perfect articles with a careless writer still produce errors.

```python
import chromadb, anthropic

col = chromadb.Client().get_or_create_collection("kb")
col.add(ids=["1", "2"], documents=[
    "The Eiffel Tower is 330 metres tall after a 2022 antenna addition.",
    "The Eiffel Tower was completed in 1889 for the World's Fair.",
])
question = "How tall is the Eiffel Tower and when was it built?"

# RETRIEVAL: returns stored text verbatim, creates nothing new
chunks = col.query(query_texts=[question], n_results=2)["documents"][0]
print(chunks)
# ['The Eiffel Tower is 330 metres tall ...', 'The Eiffel Tower was completed in 1889 ...'] (order may vary)

# GENERATION: writes a new sentence combining the evidence
resp = anthropic.Anthropic().messages.create(
    model="claude-sonnet-5-5", max_tokens=1024,
    system="Answer only from the provided context...",
    messages=[{"role": "user",
               "content": "Context:\n" + "\n".join(chunks) + f"\n\nQuestion: {question}"}],
)
print(resp.content[0].text)
# e.g. "The Eiffel Tower is 330 metres tall and was completed in 1889."
```

## Common Mistakes / Misconceptions
- **Evaluating only the final answer**, so you can't tell which stage failed.
- **Thinking a stronger LLM fixes bad retrieval.** It can't use evidence it never received.
- **Thinking retrieval "understands" the question** the way an LLM does; it matches similarity, not reasoning.
- **Using the generator's own knowledge** as a fallback silently; make it say "I don't know."

## Key Takeaways
- Retrieval finds and returns existing text; generation composes new text.
- Different models, costs, failure modes, and metrics.
- Debug by checking retrieved context first.
- RAG's strength comes from combining grounded retrieval with fluent generation.

## Related Questions
- [What is Retrieval-Augmented Generation (RAG)?](../Week-08-Retrieval-Augmented-Generation-RAG/01-What-is-Retrieval-Augmented-Generation-RAG.md)
- [What is a retrieval pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
