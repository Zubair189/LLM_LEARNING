# How do you deploy an LLM application in production?

> Week 12 – Final Project and Revision · Question 12

## Short Answer
Package the app as a **stateless API service** (e.g., FastAPI in a Docker container) behind an **API gateway/load balancer**, keep secrets in a **secret manager** injected as environment variables, and back it with managed **state stores** (Redis for cache/rate limits/sessions, a vector DB, a relational DB). Add **resilience** (timeouts, retries with backoff, fallbacks, streaming), **cost controls** (caching, model routing, token limits), **observability** (structured logs, traces, metrics, alerts), and a **CI/CD pipeline with an evaluation gate**, then roll out gradually (canary/feature flags) and keep monitoring quality, latency, and cost.

## Detailed Explanation

### End-to-end production architecture

```
 Users ─▶ CDN / Web app ─▶ API gateway (auth, TLS, rate limit, quotas)
                                 │
                                 ▼
               Load balancer ─▶ FastAPI pods ×N (Docker, autoscaled, stateless)
                                 │  /healthz /readyz, streaming responses
      ┌──────────────┬───────────┼─────────────┬──────────────────┐
      ▼              ▼           ▼             ▼                  ▼
  Redis          Vector DB    Postgres     LLM provider API    Queue + workers
  cache, rate    (RAG index)  users,       (primary model,     (ingestion,
  limit, session              history,     fallback model)     long jobs)
                              feedback
      │
 Secrets: vault/secret manager ─▶ env vars (ANTHROPIC_API_KEY, DB_URL)
 Observability: OpenTelemetry traces + JSON logs + metrics ─▶ dashboards, alerts
 CI/CD: tests ─▶ eval suite gate ─▶ build image ─▶ staging ─▶ canary 5% ─▶ 100%
```

### Production checklist

| Area | What to do |
|---|---|
| Packaging | Dockerfile, pinned dependencies, config via env vars (12-factor) |
| Serving | Async FastAPI, Uvicorn/Gunicorn workers, streaming for chat |
| Scaling | Stateless pods, horizontal autoscaling; state in Redis/DB |
| Reliability | Timeouts, retries with exponential backoff on 429/5xx, circuit breaker, fallback model, graceful error messages |
| Security | Auth, secret manager, least-privilege DB roles, guardrails, HTTPS only |
| Cost | Response caching, prompt caching, cheaper model for simple tasks, `max_tokens`, per-user budgets |
| Observability | Request IDs, traces across retrieval → LLM → tools; log tokens, latency, cost, errors |
| Quality | Offline eval suite in CI; online feedback; drift monitoring |
| Release | Version prompts like code; staging, canary, feature flags, quick rollback |
| Data | Backups, retention policy, PII handling, ingestion pipeline for RAG |

### Key metrics to alert on
- p50/p95 latency and time-to-first-token
- Error rate (by type: provider 429/5xx, timeouts, validation failures)
- Tokens and cost per request / per user / per day
- Cache hit rate
- Quality signals: eval scores, thumbs-down rate, "I don't know" rate

### Deployment options
- **Managed API model** (simplest): your service calls the provider; you deploy only your app.
- **Self-hosted model** (open weights): add GPU serving (e.g., vLLM, Ollama for dev) — you now own model scaling, latency, and upgrades.
- **Hybrid:** small local model for cheap tasks, hosted frontier model for hard ones.

## Example
**Analogy:** Opening a restaurant chain. The recipe (prompt + model) is only part of it: you also need identical kitchens (containers), a host managing the queue (gateway/rate limits), a pantry (databases/cache), health inspections (evals before release), a soft opening (canary), and a manager watching wait times and complaints (monitoring).

**Python – production-ready core of the service:**
```python
import os, time, json, logging, uuid
import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("app")
client = anthropic.AsyncAnthropic(max_retries=3, timeout=30.0)  # key from env
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
app = FastAPI()
cache: dict[str, str] = {}            # use Redis in production

class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

@app.get("/healthz")
async def health():
    return {"status": "ok"}

@app.post("/ask")
async def ask(body: Ask):
    rid, start = str(uuid.uuid4()), time.perf_counter()
    if body.question in cache:
        log.info(json.dumps({"rid": rid, "event": "cache_hit"}))
        return {"answer": cache[body.question], "cached": True}
    try:
        resp = await client.messages.create(
            model=MODEL, max_tokens=500,
            system=[{"type": "text", "text": "You are a helpful support assistant.",
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": body.question}])
    except anthropic.APIError as e:
        log.error(json.dumps({"rid": rid, "event": "llm_error", "error": type(e).__name__}))
        raise HTTPException(503, "Assistant temporarily unavailable")
    answer = resp.content[0].text
    cache[body.question] = answer
    log.info(json.dumps({"rid": rid, "event": "llm_ok", "model": MODEL,
        "latency_ms": round((time.perf_counter() - start) * 1000),
        "in_tok": resp.usage.input_tokens, "out_tok": resp.usage.output_tokens}))
    return {"answer": answer, "cached": False}

# Run: uvicorn main:app --host 0.0.0.0 --port 8000
# Log: {"rid": "...", "event": "llm_ok", "model": "claude-sonnet-5-5", "latency_ms": 1200, "in_tok": 25, "out_tok": 80}  (illustrative)
```
Note: prompt caching only kicks in above a minimum prompt length, so it pays off for long system prompts; check the provider docs. Pricing and limits change — check the provider pricing page.

## Common Mistakes / Misconceptions
- **Keeping state in process memory** (history, rate limits) — breaks with multiple pods.
- **No timeouts or retries** — one slow provider call ties up workers.
- **Shipping prompt changes without running evals** — silent quality regressions.
- **Monitoring only uptime** — an LLM app can be "up" while giving wrong or expensive answers.

## Key Takeaways
- Stateless containerized API + gateway + managed state stores.
- Resilience: timeouts, retries/backoff, fallbacks, streaming.
- Cost: caching, model routing, token limits, budgets.
- Observability of latency, errors, tokens, cost, and quality.
- CI/CD with an eval gate and gradual rollout.

## Related Questions
- [How do you design an LLM-powered application?](../Week-11-Production-LLM-Applications/01-How-do-you-design-an-LLM-powered-application.md)
- [How does FastAPI help in AI applications?](../Week-11-Production-LLM-Applications/03-How-does-FastAPI-help-in-AI-applications.md)
- [What is response caching?](../Week-11-Production-LLM-Applications/10-What-is-response-caching.md)
- [How do you reduce LLM API costs?](../Week-11-Production-LLM-Applications/12-How-do-you-reduce-LLM-API-costs.md)
- [How do you monitor latency and failures?](../Week-11-Production-LLM-Applications/16-How-do-you-monitor-latency-and-failures.md)
- [How do you handle API errors?](../Week-05-Python-for-LLM-Engineering/15-How-do-you-handle-API-errors.md)
- [What is the difference between local and cloud LLMs?](../Week-06-Working-with-LLMs-and-Model-APIs/06-What-is-the-difference-between-local-and-cloud-LLMs.md)
- [How do you secure an LLM application?](10-How-do-you-secure-an-LLM-application.md)
