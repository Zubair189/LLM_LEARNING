# What is context management?

> Week 11 – Production LLM Applications · Question 5

## Short Answer
Context management is the practice of deciding exactly what goes into the model's limited context window on each call, and in what order. That covers the system prompt, conversation history, retrieved documents, tool results and the user's question. Good context management keeps prompts relevant, within the token limit and cheap. It also stops important instructions from being diluted or pushed out.

## Detailed Explanation
The context window is a fixed token budget shared by input and output. Everything competes for it:

```
┌──────────────────── context window (e.g. N tokens) ────────────────────┐
│ system prompt │ summary of old turns │ recent turns │ retrieved chunks │
│ tool results  │ current user question │  ← reserve room for the answer │
└─────────────────────────────────────────────────────────────────────────┘
```

**Core techniques**
| Technique | What it does |
|---|---|
| Budgeting | Allocate tokens per section (e.g. 1k system, 2k history, 4k docs, 1k output) |
| Truncation / sliding window | Drop the oldest turns first |
| Summarisation | Compress old turns into a running summary |
| Retrieval (RAG) | Include only the top-k relevant chunks, not whole documents |
| Re-ranking & dedup | Keep the best, non-redundant chunks |
| Tool-result trimming | Return only needed fields from APIs/SQL, not 10,000 rows |
| Ordering | Stable content (system, docs) first, which also enables prompt caching; question last |
| Isolation | Wrap untrusted text in clear delimiters so it is treated as data |

**How it differs from conversation history:** history (Q4) is *storage* of past turns. Context management is the *selection and assembly* of everything sent on one call, history included.

## Example
**Analogy:** Packing a carry-on bag. You can't bring your whole wardrobe, so you pick what matters for this trip, roll things tightly (summaries) and leave space for souvenirs (the model's output).

```python
import anthropic

client = anthropic.Anthropic()

def fit_history(history: list[dict], system: str, question: str, budget: int) -> list[dict]:
    """Drop oldest turn pairs until the prompt fits the input-token budget."""
    msgs = history + [{"role": "user", "content": question}]
    while True:
        count = client.messages.count_tokens(
            model="claude-sonnet-5-5", system=system, messages=msgs).input_tokens
        if count <= budget or len(msgs) <= 1:
            return msgs
        msgs = msgs[2:]  # remove oldest user+assistant pair

history = [{"role": "user", "content": "long text ..." * 500},
           {"role": "assistant", "content": "ok"},
           {"role": "user", "content": "Short follow-up"},
           {"role": "assistant", "content": "Sure"}]
msgs = fit_history(history, "Be concise.", "Summarise our chat", budget=200)
print(len(msgs))
# -> 3   (the long first pair was dropped; illustrative)
```

## Common Mistakes / Misconceptions
- "Bigger context window means no need to manage context." Long prompts cost more, run slower, and models can miss details buried in the middle.
- Forgetting to reserve tokens for the output (`max_tokens`).
- Stuffing whole documents instead of relevant chunks.
- Putting changing content before stable content, which defeats prompt caching.

## Key Takeaways
- The context window is a budget. Spend it on the most relevant tokens.
- Combine truncation, summarisation and retrieval.
- Order matters, both for model attention and for caching.
- Treat untrusted context (documents, tool output) as data, not instructions.

## Related Questions
- [How do you maintain conversation history?](04-How-do-you-maintain-conversation-history.md)
- [What is a context window?](../Week-02-Introduction-to-LLMs-and-Generative-AI/09-What-is-a-context-window.md)
- [What is context injection?](../Week-08-Retrieval-Augmented-Generation-RAG/10-What-is-context-injection.md)
- [How do you reduce LLM API costs?](12-How-do-you-reduce-LLM-API-costs.md)
