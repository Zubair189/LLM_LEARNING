# Why do LLMs need external knowledge?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 2

## In One Line
An LLM only knows what it learned during training, so we must **give it extra information** to answer about new, private or very specific facts.

## Simple Explanation
During training, an LLM reads a huge amount of text and stores what it learned inside its **weights** (the millions of numbers inside the model). After training, that knowledge is frozen.

This causes some problems:

1. **Knowledge cutoff:** it doesn't know anything that happened after training ended, like last week's price change.
2. **No private data:** it has never seen your company wiki, contracts or customer records.
3. **Fuzzy memory:** it can mix up details like version numbers or dates.
4. **Made-up answers:** when it doesn't know, it may confidently invent an answer. This is called a **hallucination**.
5. **No sources:** it can't tell you where a fact came from.

**The fix:** give the model the right information **at the moment you ask the question**. This "external knowledge" can come from documents (RAG), a database, a web search or an API (another program it can call).

**Why not just retrain the model?** Retraining is slow and expensive, and it doesn't reliably store facts. Updating a document collection takes seconds.

**Why not paste every document into the prompt?** It costs more, is slower, and models can miss facts buried in the middle of a very long prompt. Picking a few relevant pieces works better.

**Link to LLMs:** this is exactly why RAG exists. LLMs are great at reasoning and writing; external knowledge makes them reliable with facts.

## Real-Life Example
A brilliant new employee who graduated last year. They think and write well, but they don't know your products, this month's prices or yesterday's policy change. Give them access to the company intranet and their answers become accurate.

## Code Example
Needs `pip install anthropic` and an API key (not run here, so the output is an example):

```python
import anthropic

question = "What is the late-return fee at Northwind Library?"
# The model can't know this, so we supply the real policy text
policy = "Northwind Library policy (2026): late returns cost $0.25 per day, max $10."

resp = anthropic.Anthropic().messages.create(
    model="claude-sonnet-5-5", max_tokens=1024,
    system="Answer only from the context. If it isn't there, say you don't know.",
    messages=[{"role": "user", "content": f"Context:\n{policy}\n\nQuestion: {question}"}],
)
print(resp.content[0].text)   # e.g. "The fee is $0.25 per day, up to $10."
```

**What happened:** the model answered correctly because we gave it the policy. Without the context it could only guess.

## Watch Out
- A bigger model does **not** fix this. Size doesn't give it your private data or today's news.
- External information can be wrong or outdated too. Keep your sources clean and current.

## Remember
- LLM knowledge is frozen at training time.
- External knowledge gives current, private, specific facts.
- RAG is the most common way to provide it.

## Related Questions
- [What is Retrieval-Augmented Generation (RAG)?](../Week-08-Retrieval-Augmented-Generation-RAG/01-What-is-Retrieval-Augmented-Generation-RAG.md)
- [How is RAG different from fine-tuning?](../Week-08-Retrieval-Augmented-Generation-RAG/03-How-is-RAG-different-from-fine-tuning.md)
- [What is hallucination in RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/12-What-is-hallucination-in-RAG.md)
