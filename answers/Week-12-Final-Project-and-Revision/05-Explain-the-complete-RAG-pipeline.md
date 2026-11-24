# Explain the complete RAG pipeline.

> Week 12 – Final Project and Revision · Question 5

## In One Line
RAG (Retrieval-Augmented Generation) means the app **first finds the relevant parts of your documents**, then gives them to the LLM so it answers **from those documents** and can cite them.

## Simple Explanation
An LLM only knows what it learned during training. It doesn't know your company's policies, your private files, or anything newer than its training data. Retraining it every time is slow and expensive. RAG solves this by looking things up at question time.

RAG has two halves:

```
Ingestion (ahead of time): Documents → Clean → Chunk → Embed → Vector DB (+ file, page, permissions)
Query (each question):     Question → Embed → Search → Filter & rerank → Prompt with chunks → LLM → Check → Answer + sources
```

**Step by step:**

**Half 1: Ingestion (done once, and again when documents change)**
1. **Load** your documents (PDF, web pages, Word files) and **clean** the text.
2. **Chunk** them: split into pieces of, say, about 500 tokens, with a little **overlap** between neighbours so sentences at the edges aren't cut off.
3. **Embed** each chunk: turn it into a vector of numbers that captures its meaning (see Q4).
4. **Store** the vectors in a **vector database**, with **metadata** like file name, page and who is allowed to see it.

**Half 2: Answering a question**

5. (Optional) **Rewrite the question** using chat history, so "what about contractors?" becomes a full question.
6. **Embed the question** with the same embedding model.
7. **Search** for the most similar chunks, e.g. the top 20.
8. **Filter** out chunks this user isn't allowed to see, and **rerank** (re-order more carefully) to keep the best 3–5.
9. **Build the prompt:** clear instructions ("answer only from the documents, cite the source, say *I don't know* if it's not there"), then the chunks, then the question.
10. **Generate** the answer with the LLM.
11. **Check** the answer: are there citations? Is it supported by the chunks?
12. **Return** the answer with its sources, and **log** which chunks were used.

**When answers are wrong, find out which half failed:**
- **Retrieval failure:** the right chunk was never found. Fix chunking, embeddings, search settings or add keyword search.
- **Generation failure:** the right chunk was in the prompt, but the answer is still wrong. Fix the prompt or the model.

This is why you should **always log which chunks were retrieved**.

**Safety:** retrieved documents are **untrusted**. They might hide a prompt-injection trick. Wrap them in tags, tell the model they are data, and check permissions *before* retrieval.

## Real-Life Example
An open-book exam. Ingestion is building a well-organised binder of notes. Retrieval is flipping to the right pages. Generation is writing your answer using those pages and noting where each fact came from. A plain LLM is a closed-book exam: it relies on memory and is more likely to make things up.

## Watch Out
- RAG **reduces** hallucinations but doesn't remove them. You still need clear instructions, citations and checks.
- Most RAG problems come from **bad chunking or retrieval**, not from the LLM. Don't only tweak the prompt.

## Remember
- **Ingestion:** load → chunk → embed → store.
- **Query:** embed → search → filter/rerank → prompt → generate → check.
- Answer **only from the documents**, cite sources, and test retrieval and generation separately.

## Related Questions
- [What is Retrieval-Augmented Generation (RAG)?](../Week-08-Retrieval-Augmented-Generation-RAG/01-What-is-Retrieval-Augmented-Generation-RAG.md)
- [What is document chunking?](../Week-08-Retrieval-Augmented-Generation-RAG/05-What-is-document-chunking.md)
- [What is chunk overlap?](../Week-08-Retrieval-Augmented-Generation-RAG/07-What-is-chunk-overlap.md)
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
- [What is an indirect prompt injection attack?](../Week-11-Production-LLM-Applications/07-What-is-an-indirect-prompt-injection-attack.md)
- [Explain embeddings and vector search.](04-Explain-embeddings-and-vector-search.md)
