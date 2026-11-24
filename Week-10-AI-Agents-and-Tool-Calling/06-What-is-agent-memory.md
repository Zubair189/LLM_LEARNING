# What is agent memory?

> Week 10 – AI Agents and Tool Calling · Question 6

## Short Answer
Agent memory is everything that lets an agent use information from earlier steps or earlier sessions. LLMs are stateless, so all memory is something *your system* stores and puts back into the prompt. It comes in two main forms. **Short-term memory** is the current context window: messages, tool calls and results. **Long-term memory** is persisted data, such as user preferences, past outcomes or documents, kept in files, databases or vector stores and retrieved when relevant.

## Detailed Explanation

```
            ┌──────────── context window (short-term) ────────────┐
 retrieve ─▶│ system prompt | retrieved memories | chat | tool I/O │─▶ LLM
            └──────────────────────────────────────────────────────┘
                                   │ write / summarize
                                   ▼
               long-term store (DB, files, vector index, memory tool)
```

| Type | Analogy | Lives in | Example |
|---|---|---|---|
| Short-term / working | RAM | The `messages` list | "The user just said the order ID is 812" |
| Long-term: semantic | Knowledge | Vector DB / KB | Company policies, facts |
| Long-term: episodic | Diary | DB of past runs | "Last week the deploy failed because of X" |
| Long-term: procedural | Habits | System prompt, skills, rules files | "Always run tests before committing" |
| User profile | Contacts card | Key-value store | "Prefers metric units" |

**Core operations:** *write* (what to save), *retrieve* (what to load now), *compact* (summarize old turns), and *forget* (TTL, deletion for privacy or GDPR).

**Why it's hard:**
- Context windows are finite and costly. 200K tokens is large but not infinite, and quality can drop with very long contexts.
- Retrieving the wrong memory pollutes reasoning.
- Memory is an attack surface. Injected text saved today can be "recalled" tomorrow (memory poisoning).

## Example
**Analogy:** a detective has a notepad on the desk for the current case (short-term) and a filing cabinet of old cases (long-term). The notepad is small, so they file notes away and pull folders out only when a case needs them.

```python
import json, pathlib

MEM = pathlib.Path("memory.json")

def load_memory(user_id: str) -> dict:
    data = json.loads(MEM.read_text()) if MEM.exists() else {}
    return data.get(user_id, {})

def save_memory(user_id: str, key: str, value: str) -> None:
    data = json.loads(MEM.read_text()) if MEM.exists() else {}
    data.setdefault(user_id, {})[key] = value
    MEM.write_text(json.dumps(data))

save_memory("u42", "units", "metric")             # long-term write
profile = load_memory("u42")                       # long-term read
system = f"Known user preferences: {json.dumps(profile)}"
messages = [{"role": "user", "content": "How warm is it in Rome?"}]  # short-term
print(system)   # -> Known user preferences: {"units": "metric"}
# pass system=system, messages=messages to client.messages.create(...)
```

## Common Mistakes / Misconceptions
- **"The model remembers me."** It only knows what is in the current request. Your app supplies the memory.
- **Stuffing the full history into every call.** This is expensive and slow, and it can degrade answers. Summarize or retrieve instead.
- **Saving everything without curation.** Noisy memory leads to wrong recall.
- **Ignoring privacy.** Long-term memory needs access control, retention limits and deletion.

## Key Takeaways
- LLMs are stateless. Agent memory is an engineering layer around them.
- Short-term memory is the context window. Long-term memory is external storage plus retrieval.
- Design write, retrieve, compact and forget policies explicitly.
- Treat stored memory as untrusted input, and protect it as user data.

## Related Questions
- [What is short-term memory?](07-What-is-short-term-memory.md)
- [What is long-term memory?](08-What-is-long-term-memory.md)
- [What is an AI agent?](01-What-is-an-AI-agent.md)
