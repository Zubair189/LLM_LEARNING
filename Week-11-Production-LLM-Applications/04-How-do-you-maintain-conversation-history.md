# How do you maintain conversation history?

> Week 11 – Production LLM Applications · Question 4

## Short Answer
LLM APIs are stateless: the model remembers nothing between calls. To hold a conversation, your application stores the messages (user and assistant turns) per session, for example in Redis or a database. It then sends the relevant history back with every new request in the `messages` list. As the conversation grows, you trim, summarise or retrieve older turns so the prompt stays within the context window and budget.

## Detailed Explanation
**Flow per turn:**
```
 user msg + session_id
        │
        ▼
 load history[session_id] from store (Redis/DB)
        │
        ▼
 build messages = trimmed_history + new user msg
        │
        ▼
 LLM call ──► assistant reply
        │
        ▼
 append (user, assistant) to store, set TTL ──► return reply
```

**Storage options**
| Store | Good for |
|---|---|
| In-memory dict | Demos and single-process tests (lost on restart) |
| Redis (with TTL) | Fast, shared across workers, auto-expiring sessions |
| SQL/NoSQL DB | Durable history, audits, analytics |

**Keeping history manageable**
- **Sliding window:** keep the last N turns.
- **Token budget:** drop the oldest turns until the history fits, e.g. 4,000 tokens.
- **Summarisation:** replace old turns with a short running summary.
- **Retrieval memory:** embed past turns and fetch only the relevant ones (long-term memory).

Rules to respect: messages alternate `user`/`assistant`, and the system prompt is passed separately via `system=` and never stored as a user turn.

## Example
**Analogy:** Phoning a support line where every agent has amnesia. The company keeps a case file, and each new agent reads the recent notes before answering you.

```python
import json, os
import anthropic, redis.asyncio as redis
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
client = anthropic.AsyncAnthropic()
r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
MAX_TURNS = 10  # keep last 10 messages

class ChatIn(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(body: ChatIn):
    key = f"chat:{body.session_id}"
    history = [json.loads(m) for m in await r.lrange(key, 0, -1)]
    messages = history[-MAX_TURNS:] + [{"role": "user", "content": body.message}]
    resp = await client.messages.create(
        model="claude-sonnet-5-5", max_tokens=512,
        system="You are a friendly assistant.", messages=messages)
    reply = resp.content[0].text
    await r.rpush(key, json.dumps({"role": "user", "content": body.message}),
                  json.dumps({"role": "assistant", "content": reply}))
    await r.expire(key, 60 * 60 * 24)  # 24h TTL
    return {"reply": reply}
# Turn 1: {"session_id":"s1","message":"My name is Ana"} -> "Nice to meet you, Ana!"
# Turn 2: {"session_id":"s1","message":"What's my name?"} -> "Your name is Ana."
```

## Common Mistakes / Misconceptions
- Believing the model "remembers" earlier calls. Only what you send is seen.
- Sending the entire history forever, so cost and latency grow every turn.
- Trimming so that the list starts with an `assistant` message or orphans a tool result.
- Storing history without per-user isolation or retention limits, which is a privacy risk.

## Key Takeaways
- Memory lives in your application, not in the model.
- Store per session, send back a trimmed or summarised view each turn.
- Redis with TTL is a common, simple production choice.
- History length directly drives token cost. Manage it deliberately.

## Related Questions
- [What is context management?](05-What-is-context-management.md)
- [What is token consumption?](11-What-is-token-consumption.md)
- [What is short-term memory?](../Week-10-AI-Agents-and-Tool-Calling/07-What-is-short-term-memory.md)
- [What is a context window?](../Week-02-Introduction-to-LLMs-and-Generative-AI/09-What-is-a-context-window.md)
