# What are GET and POST requests?

> Week 05 – Python for LLM Engineering · Question 13

## Short Answer
GET and POST are the two most common HTTP methods. **GET** *retrieves* data. Parameters go in the URL query string, there is normally no body, and it should not change anything on the server (it's "safe" and idempotent). **POST** *sends* data in the request body, usually JSON, to create something or trigger processing, so it isn't idempotent. LLM APIs use POST for generation (`POST /v1/messages`) because prompts are large, structured and sensitive, and GET for lookups such as listing available models.

## Detailed Explanation

| Aspect | GET | POST |
|---|---|---|
| Purpose | Read / fetch | Create / submit / process |
| Data location | URL query: `?limit=20` | Request body (JSON, form, file) |
| Size | Limited by URL length | Large payloads OK |
| Safe (no side effects)? | Yes, by convention | No |
| Idempotent (repeat = same effect)? | Yes | No |
| Cached by browsers/proxies? | Often | Generally not |
| Visible in logs/history | Yes, the URL is logged | Body usually not logged |
| LLM example | `GET /v1/models` | `POST /v1/messages` |

**Other methods you'll meet:** `PUT` (replace), `PATCH` (partial update), `DELETE` (remove). Together these map to CRUD in REST APIs.

**Why idempotency matters for retries:** retrying a GET is always safe. Retrying a POST could do the work twice (two charges, two records). For LLM generation a duplicate usually just costs extra tokens, but for actions like "send email" or "create order" you need care. Some APIs accept an idempotency key for this.

```
GET  /v1/models?limit=5          ──►  "Which models exist?"     (read)
POST /v1/messages  {prompt...}   ──►  "Generate a reply to this" (process)
```

## Example
**Analogy:** GET is reading the menu on the wall: you just look, and nothing changes. POST is handing the waiter your order: you send details, and something new happens in the kitchen.

```python
# pip install httpx ; requires Ollama running locally (ollama serve / ollama run llama3.2)
import httpx

BASE = "http://localhost:11434"

# GET: list locally installed models (read-only)
tags = httpx.get(f"{BASE}/api/tags", timeout=10).json()
print([m["name"] for m in tags["models"]])          # e.g. ['llama3.2:latest']

# POST: send a chat request (body = JSON)
resp = httpx.post(
    f"{BASE}/api/chat",
    json={
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "One word: capital of Japan?"}],
        "stream": False,
    },
    timeout=60,
)
print(resp.json()["message"]["content"])            # e.g. Tokyo

# GET with query parameters: httpx builds ?q=llm&limit=3 for you
r = httpx.get("https://httpbin.org/get", params={"q": "llm", "limit": 3})
print(r.json()["args"])                             # {'limit': '3', 'q': 'llm'}
```

## Common Mistakes / Misconceptions
- **Sending secrets or long prompts in GET query strings.** They are logged and length-limited.
- **Using GET for operations that change state.** Crawlers and prefetchers may trigger them by accident.
- **Thinking POST is "secure" and GET is not.** Both are only secure over HTTPS. POST just keeps data out of the URL.
- **Blindly auto-retrying non-idempotent POSTs** that have real-world side effects.

## Key Takeaways
- GET = read with URL parameters. POST = send a body to create or process.
- GET is safe and idempotent. POST is neither.
- LLM generation endpoints are POST with a JSON body. Discovery endpoints are often GET.
- Consider idempotency before retrying.

## Related Questions
- [What is an HTTP request?](12-What-is-an-HTTP-request.md)
- [How do you consume a REST API using Python?](14-How-do-you-consume-a-REST-API-using-Python.md)
- [What is Ollama?](../Week-06-Working-with-LLMs-and-Model-APIs/05-What-is-Ollama.md)
