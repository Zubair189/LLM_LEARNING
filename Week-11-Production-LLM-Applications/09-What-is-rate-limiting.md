# What is rate limiting?

> Week 11 – Production LLM Applications · Question 9

## Short Answer
Rate limiting caps how many requests, or how many tokens, a client may use in a time window, for example 20 requests per minute per user. In LLM apps it protects you from abuse, runaway costs and overload, and it keeps you under the provider's own limits. A common algorithm is the **token bucket**: each request spends a token, tokens refill at a steady rate, and requests are rejected with HTTP 429 when the bucket is empty.

## Detailed Explanation
**Two directions of rate limiting**
1. **Inbound (your limits on your users):** per user, API key or IP; often by requests *and* by LLM tokens or cost per day.
2. **Outbound (provider limits on you):** providers cap requests and tokens per minute. When you hit them you get `429`, so retry with exponential backoff and queue work.

**Common algorithms**
| Algorithm | Idea | Note |
|---|---|---|
| Fixed window | Count per minute, reset on the minute | Simple; bursty at window edges |
| Sliding window | Count over the last 60 s | Smoother, slightly more storage |
| Token bucket | Bucket of capacity C refills at R/s | Allows short bursts, enforces average rate |
| Leaky bucket | Requests drain at a fixed rate | Smooth output, queues bursts |

```
 Token bucket (capacity 5, refill 1/sec)
 [●●●●●] ─ request ─► [●●●●○] ─ request ─► ... [○○○○○] ─► 429 Too Many Requests
             ▲ refills one ● every second
```

In production, store counters in **Redis** so all workers share them, and return a `Retry-After` header.

## Example
**Analogy:** A coffee shop gives each customer a loyalty card with 5 stamps that refill one per hour. You can grab several coffees at once, but not unlimited coffee all day.

```python
import time
from fastapi import FastAPI, Depends, HTTPException, Header

app = FastAPI()

class TokenBucket:
    def __init__(self, capacity: int, refill_per_sec: float):
        self.capacity, self.rate = capacity, refill_per_sec
        self.tokens, self.updated = float(capacity), time.monotonic()

    def allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

buckets: dict[str, TokenBucket] = {}   # in-memory; use Redis across workers

async def rate_limit(x_user_id: str = Header(...)):
    bucket = buckets.setdefault(x_user_id, TokenBucket(capacity=5, refill_per_sec=0.5))
    if not bucket.allow():
        raise HTTPException(429, "Too many requests", headers={"Retry-After": "2"})

@app.post("/ask", dependencies=[Depends(rate_limit)])
async def ask():
    return {"ok": True}
# 5 rapid calls -> {"ok": true}; 6th call -> 429 {"detail": "Too many requests"}
```

## Common Mistakes / Misconceptions
- Limiting only request count. One request with a 100k-token prompt costs far more than ten small ones, so also cap tokens or spend.
- In-memory counters with multiple workers, where each worker has its own bucket. Use Redis.
- Retrying provider 429s immediately in a tight loop. Use exponential backoff with jitter.
- No clear error to the client (missing `Retry-After`).

## Key Takeaways
- Rate limiting protects budget, stability and fairness.
- Token bucket allows bursts while enforcing an average rate.
- Limit by both requests and tokens/cost; share state in Redis.
- Handle provider 429s with backoff and queues.

## Related Questions
- [What is the role of an API gateway?](02-What-is-the-role-of-an-API-gateway.md)
- [How do you manage API costs and rate limits?](../Week-06-Working-with-LLMs-and-Model-APIs/17-How-do-you-manage-API-costs-and-rate-limits.md)
- [What is token consumption?](11-What-is-token-consumption.md)
- [How do you handle API errors?](../Week-05-Python-for-LLM-Engineering/15-How-do-you-handle-API-errors.md)
