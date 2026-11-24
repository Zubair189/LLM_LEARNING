# What is a context window?

> Week 02 – Introduction to LLMs and Generative AI · Question 9

## Short Answer
The context window is the **maximum number of tokens an LLM can consider at once**. It covers everything in a single request: the system prompt, the conversation history, any documents or tool results you include, *and* the tokens the model generates in reply. Anything outside the window is invisible to the model. As of 2026, frontier models typically offer windows from about 128k up to 1M+ tokens, while smaller or local models often have less. A bigger window lets you include more, but it costs more and adds latency, and models don't always use very long contexts perfectly.

## Detailed Explanation
**What fills the window in one API call:**
```
|<------------------- context window (e.g. 200,000 tokens) ------------------->|
[ system | tools | chat history | retrieved docs | user msg ][  model output   ]
|<------------------------ input tokens ------------------->|<- max_tokens ->|
```
Rule: `input_tokens + max_tokens (output) ≤ context window`. If you exceed it, the API returns an error, or some applications silently drop old messages.

**Key facts:**
- **The model is stateless.** "Memory" in a chat only exists because the application resends the history each turn, which uses up the window.
- **Cost and latency grow with context.** You pay for every input token on every call. Prompt caching can greatly cut the cost of repeated prefixes.
- **Long contexts aren't used perfectly.** Models may pay less attention to information buried in the middle of very long inputs ("lost in the middle"). Quality on hard reasoning can drop as context grows. Put key instructions and the question in clear, prominent positions.
- **Context window ≠ max output.** Output is limited separately by `max_tokens` and by a model-specific output cap.

**Strategies when content doesn't fit (or shouldn't be sent):**
| Strategy | Idea | Week |
|---|---|---|
| RAG | Retrieve only the relevant chunks | 8 |
| Summarize history | Compress old turns into a summary | 11 |
| Sliding window / truncation | Keep only the last N turns | 11 |
| Chunk-and-map-reduce | Process a long document in pieces, then combine | 8 |
| Prompt caching | Not a size fix, but makes long, repeated prefixes cheaper and faster | 11 |

**Why there is a limit at all:** standard self-attention compute grows roughly quadratically with sequence length, the memory used by the attention cache (the KV cache) grows linearly, and models must be trained on long sequences to handle them well (Week 4).

> **Note on overlap:** Week 11's [What is context management?](../Week-11-Production-LLM-Applications/05-What-is-context-management.md) covers the *engineering techniques* for staying inside the window in production. This question defines the window itself.

## Example
**Analogy:** The context window is like a desk. You can only work with the papers spread out on it. A bigger desk holds more documents, but you still can't use papers left in the filing cabinet unless you fetch them (RAG). Piling the desk too high makes it harder to find the important page.

**Python: keep a chat history inside a token budget (sliding window):**
```python
def est_tokens(text):                       # rough: ~4 chars per token
    return max(1, len(text) // 4)

def fit_history(messages, budget):
    """Keep the most recent messages that fit within the token budget."""
    kept, used = [], 0
    for msg in reversed(messages):
        t = est_tokens(msg["content"])
        if used + t > budget:
            break
        kept.append(msg)
        used += t
    return list(reversed(kept)), used

history = [{"role": "user" if i % 2 == 0 else "assistant",
            "content": f"message {i} " + "blah " * 40} for i in range(10)]
window, max_output = 400, 150
kept, used = fit_history(history, budget=window - max_output)
print(len(kept), "of", len(history), "messages kept,", used, "tokens")
# 4 of 10 messages kept, 208 tokens
```
In a real app, use the provider's token counter (`client.messages.count_tokens(...)` in the Anthropic SDK) and make sure the kept history starts with a `user` turn.

## Common Mistakes / Misconceptions
- **"The model remembers earlier conversations."** Only what is inside the current request's context exists for it.
- **"Bigger window, so stuff everything in."** It is costly and slower, and relevant facts can get diluted. Retrieval is often better.
- **Forgetting the output tokens** when budgeting. A full input leaves no room for the answer.
- **Assuming every model has the same window.** Check the model's documentation; limits vary by model and sometimes by tier or beta feature.

## Key Takeaways
- The context window is the maximum tokens per request: input (system, history, docs, tools) plus output.
- LLMs are stateless; chat memory is just resent history that uses up the window.
- Larger windows bring higher cost and latency, and quality degrades on very long contexts.
- Manage context with RAG, summarization, truncation and caching.

## Related Questions
- [What is a token?](./06-What-is-a-token.md)
- [What is context management?](../Week-11-Production-LLM-Applications/05-What-is-context-management.md)
- [How do you maintain conversation history?](../Week-11-Production-LLM-Applications/04-How-do-you-maintain-conversation-history.md)
- [Why do we divide documents into chunks?](../Week-08-Retrieval-Augmented-Generation-RAG/06-Why-do-we-divide-documents-into-chunks.md)
