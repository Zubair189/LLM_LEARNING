# What is agent memory?

> Week 10 – AI Agents and Tool Calling · Question 6

## In One Line
Agent memory is everything that lets an agent **remember earlier steps or earlier conversations**, because the LLM itself forgets everything after each call.

## Simple Explanation
An LLM is **stateless**: each call starts fresh, like someone with no memory. If the model seems to "remember" you, that's because your program sent the earlier information back in again.

So memory is something **your system** stores and puts back into the prompt. There are two main kinds:

1. **Short-term memory:** the current conversation, meaning the messages, tool calls and tool results of this task. It lives in the **context window**, the amount of text the model can see in one call. It is fast and simple, but limited in size and gone when the session ends.
2. **Long-term memory:** information saved **outside** the model, in a file, database or search index, that survives across sessions. Examples: "user prefers vegetarian food", "project uses Python 3.12". The agent looks up the relevant bits and adds them to the prompt when needed.

How they work together:
1. During a task, everything goes into short-term memory.
2. Important facts get **saved** to long-term memory.
3. Next time, the agent **retrieves** those facts and puts them back into short-term memory.

**Why it matters:** without memory, an agent would repeat questions, forget what it already did, and lose your preferences.

## Real-Life Example
Short-term memory is the notepad on your desk while you work on today's task. Long-term memory is the filing cabinet: you put important notes there at the end of the day, and pull them out again next week.

## Code Example
A tiny agent memory with both kinds.

```python
short_term = []                 # this session's messages
long_term = {}                  # saved facts (a real system would use a file or database)

def remember_now(msg):
    short_term.append(msg)      # add to the current conversation

def save_fact(key, value):
    long_term[key] = value      # keep it for future sessions

remember_now("User: I'm vegetarian. Suggest dinner.")
save_fact("diet", "vegetarian")

short_term.clear()              # session ends: short-term memory is gone

# New session: put the saved fact back into the prompt
prompt = f"Known about user: diet={long_term['diet']}. User: Suggest lunch."
print(prompt)   # Known about user: diet=vegetarian. User: Suggest lunch.
```

**What happened:** the chat itself was wiped, but the saved fact survived and was added to the next prompt, so the agent still "remembers".

## Watch Out
- The model doesn't learn or change from memory. Memory is just text you put into the prompt.
- Saving everything is a bad idea. Too much irrelevant memory confuses the model and costs more.

## Remember
- LLMs forget between calls; your system provides memory.
- Short-term = current context window; long-term = saved outside.
- Save important facts, retrieve them when relevant.

## Related Questions
- [What is short-term memory?](07-What-is-short-term-memory.md)
- [What is long-term memory?](08-What-is-long-term-memory.md)
- [What is an AI agent?](01-What-is-an-AI-agent.md)
