# What is Retrieval-Augmented Generation (RAG)?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 1

## In One Line
RAG means: **first look up the relevant information, then give it to the LLM so it can answer using that information**.

## Simple Explanation
An LLM only knows what it learned during training. It doesn't know your company's documents or yesterday's news. RAG fixes this without retraining the model.

The name has three parts:

1. **Retrieval:** search your documents and find the few pieces (called **chunks**) most related to the question. Usually this is a vector search over embeddings.
2. **Augmented:** add those chunks to the prompt, next to the question.
3. **Generation:** the LLM writes the answer using those chunks, ideally saying which chunk each fact came from.

```
question → find similar chunks → give them to the LLM → answer
```

Before any of this, there is a setup step done once: split documents into chunks, turn them into embeddings, and store them in a vector database.

**Why is RAG so popular?**
- **Fresh facts:** update the documents, not the model.
- **Private data:** the model can use your files without being trained on them.
- **Fewer made-up answers:** the model answers from real text.
- **Sources:** you can show where each answer came from.

**Link to LLMs:** RAG is the most common way companies connect LLMs like Claude to their own knowledge.

## Real-Life Example
An **open-book exam**. The student (the LLM) is smart but hasn't memorised your company handbook. RAG hands them the right pages just before they answer, and asks them to say which page they used.

## Code Example
Needs `pip install chromadb anthropic` and an API key (not run here, so the output is an example):

```python
import chromadb, anthropic

col = chromadb.Client().get_or_create_collection("handbook")
col.add(ids=["1", "2"], documents=[
    "Employees receive 25 days of paid vacation per year.",
    "Up to 5 unused vacation days can be carried over to next year.",
])

question = "How many vacation days can I carry over?"
hits = col.query(query_texts=[question], n_results=1)["documents"][0]   # 1. Retrieval
prompt = f"Context:\n{hits[0]}\n\nQuestion: {question}"                 # 2. Augmentation

resp = anthropic.Anthropic().messages.create(                            # 3. Generation
    model="claude-sonnet-5-5", max_tokens=1024,
    system="Answer only from the context. If it isn't there, say you don't know.",
    messages=[{"role": "user", "content": prompt}],
)
print(resp.content[0].text)   # e.g. "You can carry over up to 5 unused vacation days."
```

**What happened:** Chroma found the chunk about carrying over days, we put it in the prompt, and Claude answered from it.

## Watch Out
- RAG **reduces** made-up answers but doesn't remove them completely. If retrieval finds the wrong chunk, the answer can still be wrong.
- RAG does **not** teach the model anything permanent. The knowledge is given fresh with every question.

## Remember
- RAG = retrieve → add to prompt → generate.
- It gives LLMs fresh, private knowledge without retraining.
- Good retrieval is the key to good answers.

## Related Questions
- [Why do LLMs need external knowledge?](../Week-08-Retrieval-Augmented-Generation-RAG/02-Why-do-LLMs-need-external-knowledge.md)
- [How is RAG different from fine-tuning?](../Week-08-Retrieval-Augmented-Generation-RAG/03-How-is-RAG-different-from-fine-tuning.md)
- [What is a RAG pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/11-What-is-a-RAG-pipeline.md)
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
