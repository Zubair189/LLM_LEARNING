# What is the difference between retrieval and generation?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 14

## In One Line
**Retrieval finds** existing text that matches the question; **generation writes** new text (the answer) using that found text.

## Simple Explanation
RAG has two main jobs, done by two different kinds of tools.

**Retrieval (the "librarian"):**
- Searches your stored chunks and returns the most relevant ones, **word for word**.
- Creates nothing new, so it can't invent facts.
- Uses embedding models, keyword search and rerankers.
- Fast and cheap (milliseconds).
- Can fail by **missing** the right chunk or returning unhelpful ones.

**Generation (the "writer"):**
- An LLM reads the question plus the retrieved chunks and writes a fresh, readable answer.
- **Can** invent things (hallucinate).
- Slower and more expensive (seconds, paid per token).
- Can fail by **ignoring or twisting** the chunks it was given.

```
question → RETRIEVAL → [chunk 1, chunk 2, chunk 3] → GENERATION → "Answer … [1][3]"
            librarian                                  writer
```

**Why keep them separate in your head? For debugging.** When an answer is wrong, ask one question first: **"Was the right information in the retrieved chunks?"**

1. **No** → it's a **retrieval** problem. Fix chunking, embeddings, filters or search settings.
2. **Yes** → it's a **generation** problem. Fix the prompt, the model or the citation rules.

**They also have different scores.** Retrieval is measured by whether it found the right chunks (context precision and recall). Generation is measured by whether the answer sticks to those chunks (faithfulness) and answers the question (answer relevance).

**Link to LLMs:** you need both. Retrieval alone gives raw passages the user must read; generation alone relies on the model's memory and may make things up.

## Real-Life Example
A research assistant pulls the three most useful articles from the archive (retrieval). A journalist reads them and writes a clear story (generation). A great writer with the wrong articles writes a confident but wrong story. Perfect articles with a careless writer also produce mistakes.

## Watch Out
- A stronger LLM can't fix bad retrieval. It can't use evidence it never received.
- Retrieval doesn't "understand" questions like an LLM does. It only matches similarity.

## Remember
- Retrieval = find existing text; generation = write new text.
- They fail in different ways and use different scores.
- When debugging, check the retrieved chunks first.

## Related Questions
- [What is Retrieval-Augmented Generation (RAG)?](../Week-08-Retrieval-Augmented-Generation-RAG/01-What-is-Retrieval-Augmented-Generation-RAG.md)
- [What is a retrieval pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
