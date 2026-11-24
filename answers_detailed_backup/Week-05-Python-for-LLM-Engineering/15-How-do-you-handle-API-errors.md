# How do you handle API errors?

> Week 05 – Python for LLM Engineering · Question 15

## Short Answer
First, **classify** each error. Transient errors (timeouts, connection drops, 429 rate limits, 5xx/overloaded) should be **retried with exponential backoff and jitter**, respecting any `retry-after` header. Permanent errors (400 bad request, 401 bad key, 403, 404) should **fail fast** with a clear message, because retrying won't help. Always set timeouts, cap the number of retries, log a request ID, and add a fallback (a cheaper model, a cached answer or a friendly message) so users never see a raw traceback. The Anthropic SDK already retries common transient errors (`max_retries`, default 2) and raises typed exceptions you can catch.

## Detailed Explanation

| Error | Typical cause | Action |
|---|---|---|
| Timeout / connection error | Network blip, slow response | Retry with backoff |
| 429 Too Many Requests | Rate or token limit exceeded | Wait (`retry-after`), back off, then reduce concurrency |
| 500 / 502 / 503 / 529 | Provider problem or overload | Retry with backoff; consider a fallback model |
| 400 Bad Request | Invalid parameters, prompt too long | Fix the request; don't retry |
| 401 / 403 | Missing or invalid key, no permission | Fix credentials; alert |
| 404 | Wrong URL or model id | Fix the config |

**Exponential backoff with jitter:**
```
attempt:  1     2     3     4
wait:     1s    2s    4s    8s   (+ random jitter so many clients don't retry in sync)
```

**Anthropic SDK exception classes:**
- `anthropic.APIConnectionError`: network problem or timeout (`APITimeoutError` is a subclass).
- `anthropic.RateLimitError`: HTTP 429.
- `anthropic.APIStatusError`: any other non-2xx response. It has `.status_code` and `.response`. Specific subclasses include `BadRequestError`, `AuthenticationError` and `InternalServerError`.

**LLM-specific "soft" errors** (HTTP 200 but still unusable):
- `stop_reason == "max_tokens"`: the output was truncated. Raise `max_tokens` or ask for shorter output.
- Malformed JSON or schema violations. Validate, then retry once or use forced tool output.

## Example
**Analogy:** handling API errors is like calling a busy help line. If it's engaged (429/503), wait a bit longer each time before redialling. If you dialled a wrong number (404) or don't have an account (401), redialling won't help, so fix the problem.

```python
import random
import time
import anthropic

client = anthropic.Anthropic(max_retries=0, timeout=30.0)  # disable built-in retries to show the pattern

def ask(prompt: str, attempts: int = 4) -> str:
    for attempt in range(1, attempts + 1):
        try:
            resp = client.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.stop_reason == "max_tokens":
                print("warning: answer was truncated")
            return resp.content[0].text
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            reason = type(e).__name__
        except anthropic.APIStatusError as e:
            if e.status_code < 500:          # 4xx (not 429): our fault, so don't retry
                raise
            reason = f"HTTP {e.status_code}"
        if attempt == attempts:
            break
        wait = 2 ** (attempt - 1) + random.uniform(0, 0.5)
        print(f"attempt {attempt} failed ({reason}); retrying in {wait:.1f}s")
        time.sleep(wait)
    return "Sorry, the assistant is busy right now. Please try again shortly."  # graceful fallback

print(ask("Explain exponential backoff in one sentence."))
# Usually: a one-sentence explanation.
# Under load you might first see: attempt 1 failed (RateLimitError); retrying in 1.3s
```

In production, the simplest robust option is `anthropic.Anthropic(max_retries=5)`, which handles backoff and `retry-after` for you, plus your own fallback logic. Libraries like `tenacity` help for non-SDK HTTP calls.

## Common Mistakes / Misconceptions
- **Retrying everything**, including 400/401. That wastes time and quota and hides real bugs.
- **Retrying immediately in a tight loop** with no backoff or jitter, which makes rate limiting worse.
- **Infinite retries** with no cap and no overall deadline.
- **Treating HTTP 200 as success** without checking `stop_reason` or validating the content.

## Key Takeaways
- Transient errors (timeouts, 429, 5xx): retry with exponential backoff and jitter, up to a limit.
- Permanent errors (4xx): fail fast with a clear message.
- Use the SDK's typed exceptions and built-in `max_retries`. Always set timeouts.
- Log request IDs, validate outputs, and provide a graceful fallback.

## Related Questions
- [How do Python exceptions work?](08-How-do-Python-exceptions-work.md)
- [How do you consume a REST API using Python?](14-How-do-you-consume-a-REST-API-using-Python.md)
- [How do you manage API costs and rate limits?](../Week-06-Working-with-LLMs-and-Model-APIs/17-How-do-you-manage-API-costs-and-rate-limits.md)
