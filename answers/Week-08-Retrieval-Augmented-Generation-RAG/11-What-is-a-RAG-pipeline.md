# What is a RAG pipeline?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 11

## In One Line
A RAG pipeline is the **whole system from start to finish**: preparing your documents, finding the right chunks, building the prompt, and getting the LLM's answer.

## Simple Explanation
Q1 explains the idea of RAG; this question shows the full engineering picture. The **retrieval pipeline** (Q8) is just one part inside it.

A RAG pipeline has two halves:

**1. Offline: indexing (done ahead of time)**
- Load and clean documents.
- Split them into chunks.
- Turn each chunk into an embedding.
- Store everything in a vector database.

**2. Online: answering (every time a user asks)**
- **Retrieve:** find the best chunks for the question.
- **Inject:** put those chunks into the prompt with instructions.
- **Generate:** the LLM writes the answer, citing the chunks.
- **Check:** optionally verify the answer and log it for evaluation.

```
OFFLINE:  documents → chunk → embed → vector database
ONLINE:   question → find similar chunks → give them to the LLM → answer
```

Around both halves, you **evaluate** (test with sample questions) and **monitor** (watch real usage), so you know when something breaks.

**Each step can be tested and improved on its own.** If answers are bad, check each step in order: were the chunks good? Was the right chunk found? Did the LLM use it correctly?

**Link to LLMs:** frameworks like LlamaIndex and LangChain build these pipelines for you, but it helps to understand each step yourself.

## Real-Life Example
A restaurant. Offline: buy, wash and chop ingredients into labelled boxes (indexing). Online: an order comes in, the cook grabs the right boxes (retrieve), lays them out (inject), cooks the dish (generate), and someone checks the plate before it goes out (check).

## Code Example
A minimal end-to-end pipeline. Needs `pip install chromadb anthropic` and an API key (not run here, so the output is an example):

```python
import chromadb, anthropic

# ---------- OFFLINE: index ----------
col = chromadb.PersistentClient(path="./rag_db").get_or_create_collection("shop")
col.add(ids=["returns#0", "shipping#0"],
        documents=["Items can be returned within 30 days.",
                   "Standard shipping takes 3-5 days."],
        metadatas=[{"source": "returns.md"}, {"source": "shipping.md"}])

# ---------- ONLINE: answer ----------
def answer(question):
    hits = col.query(query_texts=[question], n_results=1)["documents"][0]  # retrieve
    prompt = f"Context:\n[1] {hits[0]}\n\nQuestion: {question}"             # inject
    resp = anthropic.Anthropic().messages.create(                           # generate
        model="claude-sonnet-5-5", max_tokens=1024,
        system="Answer only from the context. Cite like [1]. "
               "If the answer is not there, say you don't know.",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text

print(answer("How long do I have to return something?"))
# e.g. "You can return items within 30 days [1]."
```

**What happened:** the offline part stored two chunks once. For each question, the online part retrieves, injects and generates.

## Watch Out
- RAG is more than "vector database + LLM". Clean ingestion, good prompts and testing matter just as much.
- Without a set of test questions, you can't tell whether a change made things better or worse.

## Remember
- RAG pipeline = offline indexing + online retrieve → inject → generate.
- The retrieval pipeline is one part of it.
- Test and improve each step separately.

## Related Questions
- [What is a retrieval pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/08-What-is-a-retrieval-pipeline.md)
- [What is document ingestion?](../Week-08-Retrieval-Augmented-Generation-RAG/04-What-is-document-ingestion.md)
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
- [What is Retrieval-Augmented Generation (RAG)?](../Week-08-Retrieval-Augmented-Generation-RAG/01-What-is-Retrieval-Augmented-Generation-RAG.md)
