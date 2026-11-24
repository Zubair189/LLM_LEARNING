# How do you monitor latency and failures?

> Week 11 – Production LLM Applications · Question 16

## Short Answer
Instrument every request and every LLM call with timing and status. Track latency percentiles (p50/p95/p99), **time-to-first-token** for streaming, error rates by type (timeouts, 429s, 5xx, validation failures) and the token counts that drive latency. Send these metrics to dashboards with alerts on SLO breaches. Handle failures with timeouts, retries with exponential backoff, fallbacks (another model or a cached answer) and circuit breakers, and log every failure with a trace ID.

## Detailed Explanation
**What to measure**
| Metric | Why |
|---|---|
| p50 / p95 / p99 end-to-end latency | Averages hide slow tails; users feel p95 |
| Time to first token (TTFT) | Perceived speed for streaming UIs |
| Per-step latency (retrieval, LLM, tools) | Finds the bottleneck |
| Error rate by class | `timeout`, `rate_limited (429)`, `provider_5xx`, `bad_output` (invalid JSON) |
| Retry and fallback counts | Rising retries signal provider trouble early |
| Tokens per request | Longer outputs mean higher latency |

**Failure handling patterns**
```
call LLM (timeout=30s)
  ├─ success ─────────────────────────────► respond, record latency
  ├─ 429 / 5xx / timeout ─► retry w/ backoff (1s, 2s, 4s + jitter, max 3)
  │        └─ still failing ─► fallback model or cached answer
  └─ circuit breaker open (many recent failures) ─► fail fast, friendly error
```

**Alerting examples:** p95 > 8 s for 5 min; error rate > 2%; 429s > 50/min; daily cost > budget.

## Example
**Analogy:** A delivery company tracks not just average delivery time but the slowest 5% of deliveries. When a road is blocked (provider outage), drivers take a backup route (fallback) instead of waiting forever.

```python
import time, logging, json
import anthropic

logging.basicConfig(level=logging.INFO, format="%(message)s")
client = anthropic.Anthropic(timeout=30.0, max_retries=2)  # SDK retries 429/5xx with backoff
latencies: list[float] = []

def ask(question: str) -> str:
    for model in ("claude-sonnet-5-5", "claude-haiku-5-5"):      # primary, then fallback
        t0 = time.perf_counter()
        try:
            r = client.messages.create(model=model, max_tokens=200,
                                       messages=[{"role": "user", "content": question}])
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)
            logging.info(json.dumps({"event": "llm_ok", "model": model, "ms": round(ms)}))
            return r.content[0].text
        except (anthropic.APITimeoutError, anthropic.RateLimitError,
                anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            logging.warning(json.dumps({"event": "llm_fail", "model": model,
                                        "error": type(e).__name__}))
    return "Sorry, the assistant is temporarily unavailable."

def p95(xs: list[float]) -> float:
    s = sorted(xs)
    return s[int(0.95 * (len(s) - 1))]

for q in ["Hi", "What is RAG?", "Define latency"]:
    ask(q)
print(f"p95={p95(latencies):.0f} ms")
# -> {"event": "llm_ok", "model": "claude-sonnet-5-5", "ms": 910} ...
# -> p95=1480 ms   (illustrative)
```
In production, record latency in a histogram metric (Prometheus/OpenTelemetry) rather than a Python list.

## Common Mistakes / Misconceptions
- Watching only the average latency. A few 30-second requests ruin the experience.
- No timeouts, so a hung provider call ties up workers indefinitely.
- Retrying non-retryable errors (400 bad request, auth errors).
- Treating "HTTP 200" as success when the output was invalid JSON or empty. Track semantic failures too.

## Key Takeaways
- Track percentiles, TTFT, per-step latency and errors by class.
- Use timeouts, backoff retries, fallbacks and circuit breakers.
- Alert on SLOs, not on individual failures.
- Count bad outputs as failures, not just HTTP errors.

## Related Questions
- [What is observability?](15-What-is-observability.md)
- [How do you handle API errors?](../Week-05-Python-for-LLM-Engineering/15-How-do-you-handle-API-errors.md)
- [What is response streaming?](../Week-06-Working-with-LLMs-and-Model-APIs/14-What-is-response-streaming.md)
- [What is rate limiting?](09-What-is-rate-limiting.md)
