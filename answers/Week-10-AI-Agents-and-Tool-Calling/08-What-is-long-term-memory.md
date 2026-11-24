# What is long-term memory?

> Week 10 – AI Agents and Tool Calling · Question 8

## In One Line
Long-term memory is information an agent **saves outside the model** (in a file or database) so it can remember it in future conversations.

## Simple Explanation
Short-term memory vanishes when a session ends. Long-term memory is how an agent remembers things for days or months, such as:
- your preferences ("likes short answers"),
- past decisions ("we chose PostgreSQL"),
- project notes or lessons learned.

It works in two steps:

1. **Write (save):** during or after a task, the agent stores a useful fact. Often it does this with a tool, like `save_memory("prefers short answers")`.
2. **Read (recall):** in a new session, the agent looks up the facts that are relevant now and adds them to the prompt.

**Where is it stored?**
- **Simple files** (like a notes file). Easy to read and edit.
- **A database**, for structured facts like user settings.
- **A vector store**, a database that finds notes by *meaning* rather than exact words. So "food I can't eat" can find "allergic to peanuts".

This is very similar to **RAG**: the agent searches its own saved notes and pastes the useful ones into the prompt.

**Link to LLMs:** the model's weights never change. Long-term memory is just saved text that gets put back into the prompt.

## Real-Life Example
A good family doctor keeps a file on you. Every visit, they read your file first ("allergic to penicillin"), and afterwards they add new notes. They don't have to remember everything in their head; the file does it.

## Code Example
Save a fact to a file in one session and read it back in the next.

```python
import json

# Session 1: save a fact to disk
memory = {"user_name": "Sam", "style": "short answers"}
with open("memory.json", "w") as f:
    json.dump(memory, f)                    # write memory to a file

# ...program closes, later a new session starts...

# Session 2: load the saved facts
with open("memory.json") as f:
    memory = json.load(f)                   # read memory back

prompt = f"User is {memory['user_name']}, prefers {memory['style']}. Question: ..."
print(prompt)  # User is Sam, prefers short answers. Question: ...
```

**What happened:** the facts survived because they were saved to a file, not kept in the model. The new session put them back into the prompt.

## Watch Out
- Memories can be wrong or out of date. Let users see, fix and delete them.
- Saved text from outside (like web pages) could contain hidden instructions. Treat stored memories as data, not commands.

## Remember
- Long-term memory lives outside the model and survives sessions.
- Write useful facts, then recall the relevant ones later.
- It works like RAG over the agent's own notes.

## Related Questions
- [What is agent memory?](06-What-is-agent-memory.md)
- [What is short-term memory?](07-What-is-short-term-memory.md)
- [Can RAG replace fine-tuning?](../Week-09-Fine-tuning-and-Model-Optimization/15-Can-RAG-replace-fine-tuning.md)
