# What is long-term memory?

> Week 10 – AI Agents and Tool Calling · Question 8

## Short Answer
Long-term memory is information an agent stores **outside** the context window so it survives across sessions. Examples are user preferences, past decisions, learned facts and project notes. It lives in a database, files, or a vector store. The agent writes to it (often through a tool) and retrieves the relevant pieces back into the prompt when needed, much like RAG over the agent's own history.

## Detailed Explanation
**Storage options:**

| Store | Good for | Retrieval |
|---|---|---|
| Key-value / SQL | Profile facts ("timezone = CET") | Exact lookup |
| Vector DB | Fuzzy recall of past conversations and notes | Embedding similarity |
| Files (e.g. `NOTES.md`, rules files) | Agent-curated notes, procedures | Read/grep through a tool |
| Knowledge graph | Entities and relationships | Graph queries |

**Lifecycle:**
```
 conversation ─▶ extract salient facts ─▶ dedupe/merge ─▶ store (+timestamp, source, user_id)
 new task     ─▶ query (by user, topic, embedding) ─▶ top-k ─▶ inject into prompt
 maintenance  ─▶ expire / correct / delete on request
```

**Who decides what to save?**
- *Application-driven*: your code extracts memories after each session, for example with an LLM summarizer.
- *Agent-driven*: the agent has a memory tool (`save_memory`, `search_memory`, or a file-based memory directory) and decides itself. Memory tools are often exposed through MCP servers.

**Risks:**
- **Stale or contradictory memories.** Store timestamps and prefer the newest.
- **Memory poisoning.** A prompt injection that gets saved can affect future sessions. Validate before writing, and keep provenance.
- **Privacy and compliance.** Scope memories per user, encrypt them, and support deletion.

## Example
**Analogy:** a doctor's patient file. The doctor doesn't remember every visit, but before your appointment they pull your file and read the relevant history: allergies and past treatments.

```python
# pip install chromadb
import chromadb

client = chromadb.Client()
mem = client.create_collection("user_memories")

# write (e.g. after a session)
mem.add(ids=["m1", "m2", "m3"],
        documents=["User is vegetarian.",
                   "User's project is a Flask app deployed on Fly.io.",
                   "User prefers concise answers."],
        metadatas=[{"user": "u42"}] * 3)

# retrieve for a new task
hits = mem.query(query_texts=["suggest a dinner recipe"], n_results=1,
                 where={"user": "u42"})
print(hits["documents"][0])   # -> ['User is vegetarian.']

system = "Relevant memories:\n- " + "\n- ".join(hits["documents"][0])
# pass system=system to client.messages.create(model="claude-sonnet-5-5", ...)
```

## Common Mistakes / Misconceptions
- **Confusing long-term memory with fine-tuning.** Memory is retrieved at runtime and is editable. Fine-tuning changes weights, which is slow and hard to undo (see [Can RAG replace fine-tuning?](../Week-09-Fine-tuning-and-Model-Optimization/15-Can-RAG-replace-fine-tuning.md)).
- **Saving raw transcripts only.** Distilled facts retrieve far better.
- **Not scoping by user or tenant.** This can leak one user's memories to another.
- **Never forgetting.** Memories go stale, so build in update and delete.

## Key Takeaways
- Long-term memory = external storage plus retrieval into the context.
- Choose the store by access pattern: exact (KV/SQL) or fuzzy (vectors).
- Curate memories: extract, dedupe, timestamp, expire.
- Treat memory as sensitive and potentially poisoned data.

## Related Questions
- [What is agent memory?](06-What-is-agent-memory.md)
- [What is short-term memory?](07-What-is-short-term-memory.md)
- [Can RAG replace fine-tuning?](../Week-09-Fine-tuning-and-Model-Optimization/15-Can-RAG-replace-fine-tuning.md)
