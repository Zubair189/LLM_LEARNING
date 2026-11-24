# What is short-term memory?

> Week 10 – AI Agents and Tool Calling · Question 7

## Short Answer
In an agent, short-term (working) memory is the information in the **current context window**: the system prompt, the conversation turns, and the tool calls and results from the ongoing task. It is what the model can "see" on this call. It is limited by the context length, and it disappears when the session ends unless you save it to long-term memory.

## Detailed Explanation
**Where it lives:** the `messages` list you resend on every API call. The model has no hidden state between calls.

**Why it fills up fast in agents.** Each step adds tokens, and the whole history is resent every time:

```
Step 1: system(1k) + user(0.2k)                            = 1.2k tokens
Step 2: + assistant tool_use(0.1k) + tool_result(3k)       = 4.3k
Step 3: + tool_use(0.1k) + tool_result(8k, a web page)     = 12.4k
...
Step 15: ~80k tokens  → cost and latency climb with every step
```

Total input tokens across N steps grow roughly as **O(N²)**, because each call resends everything before it.

**Techniques for managing it:**

| Technique | What it does |
|---|---|
| Truncation / sliding window | Keep the last K turns. Simple, but it loses early facts |
| Summarization / compaction | Replace old turns with a summary |
| Tool-result trimming | Return only the needed fields, not full HTML or JSON dumps |
| Context editing | Drop stale tool results once they've been used |
| Scratchpad / notes file | Agent writes key facts to a file (moves them to long-term memory) |
| Prompt caching | Doesn't shrink context, but makes the repeated prefix cheaper and faster |

**Quality effects.** Very long contexts can cause "lost in the middle" errors. Facts buried in the middle get used less reliably. Keep working memory lean and relevant.

## Example
**Analogy:** short-term memory is a whiteboard in a meeting room. Everything on it is visible right now. Space is limited, so you erase old notes or write a summary in the corner, and when the meeting ends someone wipes it.

```python
def compact(messages: list, keep_last: int = 6, summarizer=None) -> list:
    """Keep recent turns verbatim; summarize older ones into one message."""
    if len(messages) <= keep_last:
        return messages
    old, recent = messages[:-keep_last], messages[-keep_last:]
    summary = summarizer(old) if summarizer else f"[{len(old)} earlier messages omitted]"
    # NOTE: in tool-use loops, cut at a boundary that doesn't split a
    # tool_use / tool_result pair, or the API will reject the request.
    return [{"role": "user", "content": f"Summary of earlier conversation: {summary}"}] + recent

history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
print(len(compact(history)))   # -> 7  (1 summary + 6 recent)
```

## Common Mistakes / Misconceptions
- **Assuming the model remembers earlier calls.** If a message isn't in `messages`, the model never saw it.
- **Dumping raw tool output** such as full HTML or 10k-row results into context.
- **Breaking tool_use/tool_result pairs** when truncating, which causes API errors.
- **Equating a bigger context window with better memory.** Relevance beats volume.

## Key Takeaways
- Short-term memory = the context window, resent on every call.
- Agent loops grow context quickly, so manage it actively.
- Compaction, trimming and caching keep cost, latency and accuracy under control.
- Anything that must survive the session belongs in long-term memory.

## Related Questions
- [What is agent memory?](06-What-is-agent-memory.md)
- [What is long-term memory?](08-What-is-long-term-memory.md)
- [What is an agent execution loop?](11-What-is-an-agent-execution-loop.md)
