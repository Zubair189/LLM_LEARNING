# Week 08 – Retrieval-Augmented Generation (RAG)

This week builds on embeddings and vector databases to create systems where an LLM answers from your own documents. You'll learn why LLMs need external knowledge, how RAG compares to fine-tuning, and every stage of a RAG system: document ingestion, chunking and chunk overlap, retrieval pipelines, vector similarity retrieval, metadata filtering, context injection, and generation. You'll also study what goes wrong (hallucination despite retrieved evidence) and how to measure quality with retrieval metrics (context precision/recall, recall@k, MRR) and generation metrics (faithfulness/groundedness, answer relevance), using tools like Ragas and LLM-as-judge. A bonus question covers reranking, the single most effective upgrade to a basic RAG system. By the end you should be able to build, debug, and evaluate a small production-style RAG pipeline.

**Near-duplicate guide:** Q5 (document chunking) is *how* to split; Q6 (why chunk) is the *motivation*. Q8 (retrieval pipeline) ends with a ranked list of chunks; Q11 (RAG pipeline) is the whole system: ingestion + retrieval pipeline + context injection + generation + evaluation. Q9 (vector similarity retrieval) is the RAG-retriever view of Week 07 Q9 (vector search, the general kNN/ANN operation). Q1 defines RAG conceptually; Q11 is its engineering blueprint.

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
2. **Ingestion (20 min):** load, clean, and chunk them with a sentence-aware chunker (~400 chars, 1-sentence overlap). Store in a Chroma `PersistentClient` collection with deterministic ids (`file#n`) and metadata (`source`, `section`).
3. **Retrieval (15 min):** implement `retrieve(question, k=5, where=None)`; print the chunks for 3 questions and read them. Are they the right ones?
4. **Generation (15 min):** build a prompt with `<doc id=... source=...>` tags and call `anthropic.Anthropic().messages.create(model="claude-sonnet-5-5", ...)` with a system prompt requiring citations and "I don't know" when unsupported.
5. **Reranking (15 min):** retrieve 15, rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2`, keep 4.
6. **Evaluation (25 min):** write 12 test questions (include 3 unanswerable). For each, label the relevant chunk ids. Compute recall@5 and MRR with and without reranking, and use an LLM-as-judge to score faithfulness and answer relevance (1–5).
7. **Report (10 min):** a table of metrics per configuration plus 3 failure cases, each labelled *retrieval failure* or *generation failure*.

**Deliverable:** script/notebook plus a short results table.

## Explain Aloud
Practise explaining each in under 60 seconds, without notes:
1. **The RAG pipeline end to end:** offline (ingest → chunk → embed → store) vs online (retrieve → rerank → inject → generate), and why retrieval quality caps answer quality.
2. **RAG vs fine-tuning:** knowledge via context vs behaviour via weights, when to use each, and why fine-tuning is a poor way to add facts.
3. **Evaluating RAG:** the difference between context precision/recall and faithfulness/answer relevance, how an LLM-as-judge works, and how to tell a retrieval failure from a generation failure.
