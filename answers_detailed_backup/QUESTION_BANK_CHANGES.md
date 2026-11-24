# Suggested Updates to the Question Bank

These are the changes made to, or recommended for, `LLM_Engineering_90_Day_Question_Bank.pdf` while writing the answers. The original PDF was not modified.

## 1. Bonus questions added (13)

| Week | Added question | Why |
|---|---|---|
| 1 | What is underfitting and the bias-variance tradeoff? | Overfitting was covered, but not its counterpart |
| 2 | How are chat models trained (pretraining, instruction tuning, RLHF)? | Explains how a base model becomes an assistant |
| 3 | What is a learning rate? | Gradient descent can't be understood without it |
| 4 | What is the KV cache and why does it speed up generation? | Key to how LLM inference performs in practice |
| 5 | What are async and await, and why do they matter for LLM apps? | Concurrent API calls and FastAPI depend on it |
| 5 | What are virtual environments and how do you manage dependencies? | Practical setup that every exercise needs |
| 6 | What is few-shot prompting? | A core prompt-engineering technique that was missing |
| 7 | What is hybrid search (keyword + vector)? | The default in production retrieval |
| 8 | What is reranking and why does it improve RAG? | The biggest single gain in RAG quality |
| 9 | What is preference tuning (RLHF and DPO)? | Completes the fine-tuning picture beyond SFT |
| 10 | What is the Model Context Protocol (MCP)? | The standard way to connect tools to agents |
| 11 | What are guardrails in LLM applications? | Brings safety controls together in one place |
| 12 | Capstone: design an end-to-end document Q&A assistant | A concrete blueprint for the final project |

## 2. Duplicates and overlaps

Each answer file explains its scope compared with the overlapping question.

| Overlap | How it's resolved |
|---|---|
| W1 Q10 = W9 Q5 "What is a training dataset?" (exact duplicate) | W1 covers the general ML idea; W9 covers fine-tuning datasets. **Rename W9 Q5.** |
| W10 Q3 tool calling vs Q4 function calling | Mostly synonyms; the nuance is explained |
| W7 Q1 embedding vs Q3 vector representation | Q3 is about sparse vs dense representations |
| W7 Q9 vs W8 Q9 (vector search vs vector similarity retrieval) | W8 focuses on its role as the retriever in RAG |
| W8 Q8 retrieval pipeline vs Q11 RAG pipeline; Q5 vs Q6 chunking | Each is scoped in its answer |
| W4 Q3 vs Q15 (why Transformers matter) | Historical impact vs why the mechanism suits language |
| W6 Q15 structured output vs Q16 JSON mode | JSON mode doesn't guarantee schema-valid output |
| W11 Q6 vs Q7 (prompt injection) | Direct vs indirect injection |
| W11 Q11 vs Q12 (tokens vs cost) | Measuring tokens vs cutting cost |
| W11 Q13 vs W12 Q11 (evaluation) | Evaluation as a discipline vs a rubric for one response |

## 3. Suggested rewording

| Q | Original | Suggested |
|---|---|---|
| W1 Q11 | What is the difference between training and testing data? | What are training, validation and test sets, and why must they be kept separate? |
| W2 Q5 | What is a language model? | Move it before Q2, or: How is an LLM different from earlier language models? |
| W2 Q8 | Why do LLMs use tokens instead of complete words? | Why do LLMs use subword tokens instead of whole words or characters? |
| W2 Q10 | What are model parameters? | What are model parameters, and what does a size like "70B" tell you? |
| W3 Q15 | Why do neural networks require optimization? | Why must weights be learned by iterative optimization rather than computed directly? |
| W4 Q7 | What is an attention score? | What is the difference between an attention score and an attention weight? |
| W4 Q9 | What are positional encodings? | Why do Transformers need position information, and how do sinusoidal encodings and RoPE provide it? |
| W5 Q15 | How do you handle API errors? | How do you handle API errors, and which ones should you retry? |
| W6 Q6 | What is the difference between local and cloud LLMs? | What are the trade-offs between running an LLM locally and calling a cloud API? |
| W6 Q16 | What is JSON mode? | What is JSON mode, and why isn't it enough to guarantee schema-valid output? |
| W7 Q15 | How do you retrieve similar documents using embeddings? | Walk through building semantic document search with embeddings, end to end. |
| W8 Q5 | What is document chunking? | What chunking strategies exist, and how do you choose chunk size? |
| W8 Q12 | What is hallucination in RAG? | Why can a RAG system still hallucinate, and how do you detect and reduce it? |
| W9 Q5 | What is a training dataset? | What makes a good fine-tuning dataset (format, size, quality, splits)? |
| W9 Q15 | Can RAG replace fine-tuning? | When should you use RAG, fine-tuning, or both? |
| W10 Q12 | What happens when a tool call fails? | How should an agent handle a failed tool call? |
| W11 Q13 | What is LLM evaluation? | How do you set up offline and online evaluation for an LLM application? |
| W12 Q1 | Explain the complete lifecycle of an LLM. | Explain the lifecycle of an LLM, from pretraining to production use. |
| W12 Q3 | What happens when a user sends a question to an LLM? | What happens end to end when a user sends a question to an LLM-powered app? |
| W12 Q12 | How do you deploy an LLM application in production? | How do you deploy and operate an API-based LLM application in production? |

## 4. Other notes
- The PDF says "160+ questions"; the actual count is **178** (now 191 with the bonus questions).
- Week 11 has 18 questions, more than any other week. Consider moving Q13–Q16 (evaluation and observability) into a separate topic.
