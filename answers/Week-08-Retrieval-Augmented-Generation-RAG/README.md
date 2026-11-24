# Week 08 – Retrieval-Augmented Generation (RAG)

This week uses embeddings and vector databases to build systems where an LLM answers questions **from your own documents**. This is called RAG. You'll learn why LLMs need outside knowledge, how RAG compares to fine-tuning, and every step of a RAG system: loading documents (ingestion), cutting them into chunks, overlap, finding the right chunks (retrieval), metadata filters, putting chunks into the prompt (context injection), and writing the answer (generation). You'll also see what goes wrong (hallucination, even when evidence is given) and how to measure quality: did we find the right chunks, and is the answer faithful to them? A bonus question covers reranking, one of the best upgrades for a basic RAG system. By the end you should be able to build, debug and test a small RAG app.

**Similar-looking questions:** Q5 (document chunking) is *how* to split; Q6 is *why* we split. Q8 (retrieval pipeline) ends with a list of chunks; Q11 (RAG pipeline) is the whole system. Q9 shows how RAG uses Week 07 Q9 (vector search). Q1 is the idea of RAG; Q11 is how it is built.

## Questions
1. [What is Retrieval-Augmented Generation (RAG)?](01-What-is-Retrieval-Augmented-Generation-RAG.md)
2. [Why do LLMs need external knowledge?](02-Why-do-LLMs-need-external-knowledge.md)
3. [How is RAG different from fine-tuning?](03-How-is-RAG-different-from-fine-tuning.md)
4. [What is document ingestion?](04-What-is-document-ingestion.md)
5. [What is document chunking?](05-What-is-document-chunking.md)
6. [Why do we divide documents into chunks?](06-Why-do-we-divide-documents-into-chunks.md)
7. [What is chunk overlap?](07-What-is-chunk-overlap.md)
8. [What is a retrieval pipeline?](08-What-is-a-retrieval-pipeline.md)
9. [How does vector similarity retrieval work?](09-How-does-vector-similarity-retrieval-work.md)
10. [What is context injection?](10-What-is-context-injection.md)
11. [What is a RAG pipeline?](11-What-is-a-RAG-pipeline.md)
12. [What is hallucination in RAG?](12-What-is-hallucination-in-RAG.md)
13. [What is metadata filtering?](13-What-is-metadata-filtering.md)
14. [What is the difference between retrieval and generation?](14-What-is-the-difference-between-retrieval-and-generation.md)
15. [How can we evaluate RAG answer quality?](15-How-can-we-evaluate-RAG-answer-quality.md)
16. **BONUS:** [What is reranking and why does it improve RAG?](16-BONUS-What-is-reranking-and-why-does-it-improve-RAG.md)

## Practical Exercise (1–2 hours): "Ask My Docs" Mini-RAG with an Eval Report
Setup: `pip install chromadb sentence-transformers anthropic` and set `ANTHROPIC_API_KEY`.

1. **Corpus (10 min):** pick 5–10 Markdown/text documents you know well (course notes, a project README, a public policy page).
2. **Ingestion (20 min):** load, clean and chunk them by sentences (~400 characters, 1 sentence of overlap). Store them in a Chroma `PersistentClient` collection with stable ids (`file#n`) and metadata (`source`, `section`).
3. **Retrieval (15 min):** write `retrieve(question, k=5, where=None)`. Print the chunks for 3 questions and read them. Are they the right ones?
4. **Generation (15 min):** build a prompt with `<doc id=... source=...>` tags and call `anthropic.Anthropic().messages.create(model="claude-sonnet-5-5", ...)`. In the system prompt, ask for citations and for "I don't know" when the answer isn't in the context.
5. **Reranking (15 min):** retrieve 15 chunks, reorder them with `cross-encoder/ms-marco-MiniLM-L-6-v2`, and keep the best 4.
6. **Evaluation (25 min):** write 12 test questions (3 of them with no answer in your documents). For each, note which chunk ids are relevant. Compute recall@5 and MRR with and without reranking, and use an LLM-as-judge to score faithfulness and answer relevance (1–5).
7. **Report (10 min):** a small table of scores for each setup, plus 3 failed questions, each labelled *retrieval failure* or *generation failure*.

**Deliverable:** a script or notebook, plus a short results table.

## Explain Aloud
Practise explaining each one in under 60 seconds, in simple words, without notes:
1. **The RAG pipeline from start to finish:** the offline part (load → chunk → embed → store) and the online part (retrieve → rerank → add to prompt → generate), and why good retrieval is needed for good answers.
2. **RAG vs fine-tuning:** RAG adds knowledge in the prompt, fine-tuning changes behaviour in the model; when to use each, and why fine-tuning is a poor way to add facts.
3. **Evaluating RAG:** scores for retrieval (context precision/recall) vs scores for the answer (faithfulness, answer relevance), how an LLM-as-judge works, and how to tell a retrieval failure from a generation failure.
