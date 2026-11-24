# Week 12 – Final Project and Revision

Week 12 is the capstone and revision week. Each answer is cumulative: it pulls together the whole course, from how an LLM is trained and how a Transformer works (Weeks 1–4), through calling models from Python (Weeks 5–6), embeddings, RAG, fine-tuning, and agents (Weeks 7–10), to securing, evaluating, and deploying real applications (Week 11). Every answer has an end-to-end diagram and links back to the earlier questions it builds on, so you can use this week as a map for revising the course. The bonus capstone gives you a full blueprint for the final project, a document Q&A assistant.

## Questions
1. [Explain the complete lifecycle of an LLM.](01-Explain-the-complete-lifecycle-of-an-LLM.md)
2. [Explain Transformer architecture.](02-Explain-Transformer-architecture.md)
3. [What happens when a user sends a question to an LLM?](03-What-happens-when-a-user-sends-a-question-to-an-LLM.md)
4. [Explain embeddings and vector search.](04-Explain-embeddings-and-vector-search.md)
5. [Explain the complete RAG pipeline.](05-Explain-the-complete-RAG-pipeline.md)
6. [Compare RAG and fine-tuning.](06-Compare-RAG-and-fine-tuning.md)
7. [Explain LoRA and QLoRA.](07-Explain-LoRA-and-QLoRA.md)
8. [Explain AI agent architecture.](08-Explain-AI-agent-architecture.md)
9. [How does function calling work?](09-How-does-function-calling-work.md)
10. [How do you secure an LLM application?](10-How-do-you-secure-an-LLM-application.md)
11. [How do you evaluate response quality?](11-How-do-you-evaluate-response-quality.md) (compare with Week 11 Q13, [What is LLM evaluation?](../Week-11-Production-LLM-Applications/13-What-is-LLM-evaluation.md))
12. [How do you deploy an LLM application in production?](12-How-do-you-deploy-an-LLM-application-in-production.md)
13. **BONUS:** [Capstone: design an end-to-end document Q&A assistant](13-BONUS-Capstone-design-an-end-to-end-document-QA-assistant.md)

## Practical Exercise
Build milestones 1–4 of the [capstone blueprint](13-BONUS-Capstone-design-an-end-to-end-document-QA-assistant.md):
1. Write a FastAPI service with `POST /documents` (upload a PDF or Markdown file, chunk it into about 500 tokens with 50 overlap, and store the chunks with `file` and `page` metadata in ChromaDB) and `POST /chat` (retrieve the top 4 chunks, call `claude-sonnet-5-5` with a grounded system prompt, and return the answer with `[file p.N]` citations). Read the API key from `ANTHROPIC_API_KEY`.
2. Ingest 3 documents you know well, such as a course syllabus, a product manual, or an employee handbook.
3. Write `evals/dataset.jsonl` with 20 questions: 12 answerable, 4 that need two chunks, and 4 that the documents can't answer.
4. Write `run_evals.py` to report the **retrieval hit rate@4**, **citation validity**, **refusal accuracy** on the unanswerable questions, and an **LLM-as-judge faithfulness** score.
5. Change one thing (chunk size 500 → 250, or top-k 4 → 8) and re-run. In a short README, write down which metric changed and why.

## Explain Aloud
Practice explaining each of these out loud in 2 minutes, without notes, using one diagram:
1. **The request path.** Explain what happens from the moment a user hits "send" until the streamed answer appears. Cover auth, rate limits, retrieval, prompt assembly, tokenization, token-by-token generation, output checks, and logging.
2. **RAG vs fine-tuning vs prompting.** Explain when you'd use each, why RAG is the right tool for knowledge and fine-tuning is the right tool for behavior, and how LoRA/QLoRA make fine-tuning affordable.
3. **Defense in depth for agents.** Explain why the model is not a security boundary, and how least-privilege tools, read-only SQL guards, human approval, and evaluation together keep an LLM application safe and trustworthy.
