# What is hallucination in RAG?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 12

## In One Line
A hallucination in RAG is when the LLM's answer says something that is **not supported by the documents it was given**.

## Simple Explanation
A **hallucination** is a confident-sounding answer that is made up or wrong. RAG gives the model real documents to reduce this, but it can still happen.

**Common causes:**

1. **Retrieval missed:** the right chunk wasn't found, so the model guessed from memory.
2. **Misreading:** the right chunk was there, but the model changed its meaning. The text says "up to 5 days"; the answer says "5 days guaranteed".
3. **Mixing up chunks:** it combines facts from different chunks wrongly, e.g. Plan A's price with Plan B's features.
4. **Adding extras:** it invents believable details, like a phone number that isn't in the text.
5. **Bad sources:** the documents themselves are old or contradict each other.

**Faithful vs correct:** an answer is **faithful** if every claim is supported by the given context. It can be faithful but still wrong if the document is outdated.

**How to reduce hallucinations:**
- **Better retrieval:** good chunks, hybrid search, reranking.
- **Strict prompt:** "Answer only from the context. If it's not there, say you don't know."
- **Require citations** for each claim.
- **Minimum score:** if nothing relevant is found, don't ask the LLM to answer.
- **Check the answer:** use a second LLM call (an "LLM-as-judge") to check each claim against the context.

**Link to LLMs:** "faithfulness" (or "groundedness") is the standard score for measuring this.

## Real-Life Example
An open-book exam where a student can't find the right page, so writes a confident guess. Or finds the page but copies it wrongly. The book was there, but the answer didn't really come from it.

## Code Example
An LLM-as-judge checking an answer. Needs `pip install anthropic` and an API key (not run here, so the output is an example):

```python
import anthropic

context = "Our warranty covers manufacturing defects for 12 months. Accidental damage is not covered."
answer = "The warranty lasts 12 months and covers accidental damage like drops."

resp = anthropic.Anthropic().messages.create(
    model="claude-sonnet-5-5", max_tokens=1024,
    system="You are a strict fact-checker. Judge only against the provided context.",
    messages=[{"role": "user", "content":
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
        "Mark each claim in the answer SUPPORTED or UNSUPPORTED."}],
)
print(resp.content[0].text)
# e.g.
# 1. Warranty lasts 12 months: SUPPORTED
# 2. Covers accidental damage: UNSUPPORTED (the context says it is not covered)
```

**What happened:** a second LLM call checked each claim against the context and caught the made-up part.

## Watch Out
- "RAG solves hallucination" is false. It **reduces** it.
- When an answer is wrong, check the **retrieved chunks first**. Often the problem is retrieval, not the LLM.

## Remember
- RAG hallucination = claims not backed by the given context.
- Causes: missed retrieval, misreading, mixing, inventing, bad sources.
- Fix with better retrieval, strict prompts, citations and checking.

## Related Questions
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
- [What is context injection?](../Week-08-Retrieval-Augmented-Generation-RAG/10-What-is-context-injection.md)
- [What is reranking and why does it improve RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/16-BONUS-What-is-reranking-and-why-does-it-improve-RAG.md)
