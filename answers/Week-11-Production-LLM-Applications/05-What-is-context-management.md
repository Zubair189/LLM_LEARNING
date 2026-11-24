# What is context management?

> Week 11 – Production LLM Applications · Question 5

## In One Line
Context management means **choosing what to put into the prompt** on each call, so the most useful information fits inside the model's limit.

## Simple Explanation
Every model has a **context window**: a maximum number of tokens (small pieces of text) it can read in one call. Many things compete for that space:

- the **system prompt** (the rules for the model),
- the **conversation history**,
- **retrieved documents** (from RAG),
- **tool results** (data from APIs or databases),
- the user's **question**,
- and room for the **answer** the model will write.

Context management is deciding what goes in, what gets cut and in what order. Common tricks:

1. **Budget:** decide roughly how many tokens each part may use.
2. **Trim:** drop the oldest chat messages first.
3. **Summarise:** replace old messages with a short summary.
4. **Retrieve only what's relevant:** send the best 3–5 chunks, not whole documents.
5. **Trim tool results:** send only the fields you need, not 10,000 rows.

**How is this different from conversation history (Q4)?** History is *storing* past messages. Context management is *choosing* what to send on one call, and history is just one of the things you choose from.

## Real-Life Example
Packing a small carry-on bag. You can't bring your whole wardrobe, so you pick what you need for this trip, fold things tightly (summaries), and leave some space for souvenirs (the model's answer).

## Code Example
Fit the most useful pieces into a small budget. Here we pretend 1 word ≈ 1 token.

```python
def tokens(text):                          # rough guess: 1 word ~ 1 token
    return len(text.split())

def build_context(system, history, docs, question, budget):
    used = tokens(system) + tokens(question)       # must-haves go in first
    kept_docs = []
    for d in docs:                                 # docs sorted: most useful first
        if used + tokens(d) <= budget:
            kept_docs.append(d)
            used += tokens(d)
    kept_history = []
    for msg in reversed(history):                  # walk from the newest message back
        if used + tokens(msg) > budget:
            break                                  # stop when the budget is full
        kept_history.insert(0, msg)
        used += tokens(msg)
    return kept_docs, kept_history, used

docs = ["Refunds are allowed within 30 days.",
        "Our shop opened in 1998 in a small town near the river."]
history = ["Hi there", "Hello! How can I help?", "I bought shoes last week"]
print(build_context("Answer briefly.", history, docs, "Can I get a refund?", budget=20))
# Output: (['Refunds are allowed within 30 days.'], ['I bought shoes last week'], 18)
```

**What happened:** the useful refund document and the newest message fit in. The long, unhelpful shop-history document and the old greetings were left out.

## Watch Out
- A bigger context window doesn't mean "send everything". Long prompts cost more, run slower, and the model can miss details in the middle.
- Always leave room for the answer (`max_tokens`).

## Remember
- The context window is a **budget**. Spend it on the most useful text.
- Use trimming, summaries and retrieval together.
- Treat documents and tool results as **data**, not instructions.

## Related Questions
- [How do you maintain conversation history?](04-How-do-you-maintain-conversation-history.md)
- [What is a context window?](../Week-02-Introduction-to-LLMs-and-Generative-AI/09-What-is-a-context-window.md)
- [What is context injection?](../Week-08-Retrieval-Augmented-Generation-RAG/10-What-is-context-injection.md)
- [How do you reduce LLM API costs?](12-How-do-you-reduce-LLM-API-costs.md)
