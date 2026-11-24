# What is short-term memory?

> Week 10 – AI Agents and Tool Calling · Question 7

## In One Line
Short-term memory is **everything the model can see right now** in the current conversation: your instructions, the messages so far, and the tool results.

## Simple Explanation
Each time an agent calls the LLM, it sends a bundle of text. That bundle is the model's short-term (or "working") memory. It usually contains:
- the **system prompt** (the agent's instructions),
- the **conversation so far** (user and assistant messages),
- the **tool calls and their results** from this task.

This all fits inside the **context window**, the maximum amount of text the model can read in one call. It is measured in **tokens** (small pieces of words).

Three things to know:
1. **It has a size limit.** Long tasks with big tool results can fill it up.
2. **It costs money and time.** Every call re-sends the whole bundle, so more text means slower and pricier calls.
3. **It disappears** when the session ends, unless you save important parts to long-term memory.

**What to do when it gets full:**
- **Trim:** keep only the most recent messages.
- **Summarise:** replace old messages with a short summary.
- **Shorten tool results:** return only what's needed, not a whole web page.
- **Save to long-term memory:** store key facts elsewhere and fetch them later.

## Real-Life Example
It's like a whiteboard during a meeting. Everyone can see what's on it right now, but it has limited space. When it fills up, you wipe old parts or write a short summary in the corner. At the end of the meeting, it gets erased unless someone takes a photo (long-term memory).

## Code Example
Keep only the last few messages so the memory doesn't overflow.

```python
MAX_MESSAGES = 3                         # pretend our context fits only 3 messages

history = []
for msg in ["Hi", "Book a flight", "To Rome", "On Friday", "Window seat"]:
    history.append(msg)                  # add the new message
    if len(history) > MAX_MESSAGES:
        history = history[-MAX_MESSAGES:]  # drop the oldest ones

print(history)   # ['To Rome', 'On Friday', 'Window seat']
```

**What happened:** only the newest 3 messages were kept. Notice the problem: "Book a flight" was dropped. That's why real agents often **summarise** old messages instead of just deleting them.

## Watch Out
- A bigger context window doesn't mean perfect memory. Models can miss details buried in very long contexts.
- Trimming blindly can delete the key instruction. Summarise or pin important facts.

## Remember
- Short-term memory = what's in the context window now.
- It's limited, re-sent every call, and lost at the end.
- Trim, summarise, or save important parts to long-term memory.

## Related Questions
- [What is agent memory?](06-What-is-agent-memory.md)
- [What is long-term memory?](08-What-is-long-term-memory.md)
- [What is an agent execution loop?](11-What-is-an-agent-execution-loop.md)
