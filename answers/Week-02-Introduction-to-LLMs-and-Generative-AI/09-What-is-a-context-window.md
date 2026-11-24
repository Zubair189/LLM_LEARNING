# What is a context window?

> Week 02 – Introduction to LLMs and Generative AI · Question 9

## In One Line
The context window is the **maximum amount of text (in tokens) an LLM can look at in one go**, including both your input and its reply.

## Simple Explanation
Every time you send a request, the model can only "see" what fits inside its window. That includes:

1. The **system prompt** (your instructions to the model).
2. The **chat history** so far.
3. Any **documents** or tool results you add.
4. The model's **reply**.

Simple rule: **input tokens + reply tokens ≤ context window**. Anything outside the window is invisible to the model.

Important facts:

- **The model has no memory.** A chat "remembers" only because the app resends the whole history every turn, which fills up the window.
- **Bigger windows cost more.** You pay for every input token on every call, and long inputs are slower.
- **Long inputs aren't used perfectly.** Details buried in the middle of a very long text can get missed.
- The reply has its own separate limit, set with `max_tokens`.

Today's large models often have windows from about 128,000 to over 1 million tokens. Smaller or local models usually have less.

**What to do when things don't fit:**
- Keep only the **latest messages**.
- **Summarise** older messages.
- Use **RAG** (Retrieval-Augmented Generation): fetch only the relevant parts of your documents.

Week 11's question on context management covers these techniques in depth. This one just explains what the window is.

## Real-Life Example
The context window is like a desk. You can only work with the papers on it. A bigger desk holds more, but papers in the filing cabinet are useless until you fetch them. Pile the desk too high and the important page gets lost.

## Code Example
Keep only the newest chat messages that fit in a token budget.

```python
def estimate_tokens(text):
    return max(1, len(text) // 4)             # rough: 4 characters per token

def fit_history(messages, budget):
    kept, used = [], 0
    for msg in reversed(messages):            # start from the newest message
        t = estimate_tokens(msg)
        if used + t > budget:                 # no more room? stop
            break
        kept.insert(0, msg)                   # keep it (in original order)
        used += t
    return kept, used

history = [f"message {i} " + "blah " * 40 for i in range(10)]   # 10 old messages
window, max_output = 400, 150                 # leave 150 tokens for the reply
kept, used = fit_history(history, budget=window - max_output)

print(len(kept), "of", len(history), "messages kept,", used, "tokens")
print(kept[0][:10])
# Output:
# 4 of 10 messages kept, 208 tokens
# message 6
```

**What happened:** only the 4 newest messages fit, while leaving room for the reply. The older 6 were dropped, so the model will not "remember" them.

## Watch Out
- **Don't forget the reply.** If your input fills the window, there's no room left for the answer.
- **Bigger isn't always better.** Stuffing everything in costs more and can bury the important facts.

## Remember
- Context window = everything the model can see in one request.
- Input + output must fit inside it.
- The model only "remembers" what you resend.

## Related Questions
- [What is a token?](./06-What-is-a-token.md)
- [What is context management?](../Week-11-Production-LLM-Applications/05-What-is-context-management.md)
- [How do you maintain conversation history?](../Week-11-Production-LLM-Applications/04-How-do-you-maintain-conversation-history.md)
- [Why do we divide documents into chunks?](../Week-08-Retrieval-Augmented-Generation-RAG/06-Why-do-we-divide-documents-into-chunks.md)
