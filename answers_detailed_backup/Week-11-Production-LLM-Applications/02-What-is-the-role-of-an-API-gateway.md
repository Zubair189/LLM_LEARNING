# What is the role of an API gateway?

> Week 11 – Production LLM Applications · Question 2

## Short Answer
An API gateway is the single entry point that sits between clients and your backend services. It handles cross-cutting concerns centrally: authentication, rate limiting, routing, TLS, request-size limits, logging and sometimes caching. In LLM apps it also protects your expensive model endpoints and your provider API keys. Clients never talk to the LLM provider directly.

## Detailed Explanation
Without a gateway, every service must implement auth, throttling and logging itself, and usually does so inconsistently. A gateway centralises them:

| Responsibility | Why it matters for LLM apps |
|---|---|
| Authentication / authorization | Only known users or apps can spend your tokens |
| Rate limiting & quotas | Stops abuse and runaway costs; enforces per-user/token budgets |
| Routing | `/chat` → chat service, `/search` → RAG service; A/B routing between model versions |
| Request validation & size limits | Rejects huge payloads before they become huge prompts |
| TLS termination | Encrypts traffic in one place |
| Logging / tracing | Adds a request ID so one call can be followed end-to-end |
| Caching | Returns repeated answers without another model call |
| Secret isolation | Provider keys live behind the gateway, never in the browser or mobile app |

```
 Browser / Mobile / Partner
            │  (user JWT or app key)
            ▼
   ┌──────────────────────┐
   │      API GATEWAY      │  auth · rate limit · routing · logging
   └─────────┬─────────────┘
     ┌───────┼──────────┐
     ▼       ▼          ▼
  chat-svc  rag-svc   admin-svc
     │       │
     └───┬───┘
         ▼
  LLM provider (ANTHROPIC_API_KEY stored server-side only)
```

Common gateways include Kong, NGINX, Envoy, AWS API Gateway and Azure API Management. There are also specialised "AI gateways" that add token-based quotas, model fallback and prompt logging.

## Example
**Analogy:** A hotel reception desk. Guests don't wander into the kitchen or the manager's office. Reception checks who you are, sends you to the right place and keeps a log.

A tiny gateway-style middleware in FastAPI (a real deployment would use a dedicated gateway):

```python
import os, time, uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
VALID_KEYS = set(os.environ.get("CLIENT_API_KEYS", "demo-key").split(","))
MAX_BODY = 10_000  # bytes

@app.middleware("http")
async def gateway(request: Request, call_next):
    if request.headers.get("x-api-key") not in VALID_KEYS:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if int(request.headers.get("content-length", 0)) > MAX_BODY:
        return JSONResponse({"error": "payload too large"}, status_code=413)
    rid = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    print({"rid": rid, "path": request.url.path,
           "ms": round((time.perf_counter() - start) * 1000)})
    return response

@app.get("/health")
async def health():
    return {"ok": True}
# curl -H "x-api-key: demo-key" localhost:8000/health -> {"ok": true}
# without key -> 401 {"error": "unauthorized"}
```

## Common Mistakes / Misconceptions
- Calling the LLM provider directly from front-end code, which exposes the API key.
- Assuming the gateway makes the app secure. It does not inspect prompt *content* for injection; guardrails still do that.
- Rate limiting by request count only, when LLM cost scales with tokens.
- Letting the gateway become a single point of failure, with no redundancy.

## Key Takeaways
- A gateway is the single, controlled front door to your services.
- It centralises auth, rate limiting, routing, logging and size limits.
- For LLM apps it protects keys and budgets.
- It complements application-level guardrails; it does not replace them.

## Related Questions
- [What is rate limiting?](09-What-is-rate-limiting.md)
- [How do you protect API keys?](08-How-do-you-protect-API-keys.md)
- [How do you design an LLM-powered application?](01-How-do-you-design-an-LLM-powered-application.md)
- [What is an API key?](../Week-06-Working-with-LLMs-and-Model-APIs/08-What-is-an-API-key.md)
