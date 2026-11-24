# What is rate limiting?

> Week 11 – Production LLM Applications · Question 9

## In One Line
Rate limiting puts a **cap on how many requests** someone can make in a period of time, such as 20 per minute.

## Simple Explanation
Every LLM call costs money and uses server power. Without a limit, one buggy script or one abusive user could send thousands of requests and run up a huge bill.

Rate limiting works in **two directions**:
1. **Your limits on your users:** e.g. "each user may send 20 requests a minute". When they go over, you reply with error **429 Too Many Requests**.
2. **The provider's limits on you:** the LLM provider also caps how much you can send. If you hit their 429, **wait a little and try again** (wait longer each time; this is called "backoff").

**A popular method: the token bucket.**
(Here "token" means a permission ticket, not a text token.)

1. Each user has a bucket holding, say, 3 tickets.
2. Each request uses 1 ticket.
3. Tickets refill slowly, e.g. 1 per second.
4. Empty bucket = request refused.

This allows a short burst of requests but keeps the average rate under control.

```
[●●●] → request → [●●○] → request → [●○○] → request → [○○○] → next request: 429!
           ↑ one ● comes back every second
```

In a real app with several servers, the counts are stored in **Redis** so all servers share the same bucket.

## Real-Life Example
A coffee shop gives you a card with 3 free-coffee stamps, and one stamp comes back every hour. You can grab a few coffees at once, but you can't drink unlimited coffee all day.

## Code Example
A token bucket with 3 tickets, refilling 1 per second. We pass in the time so the result is easy to follow.

```python
class TokenBucket:
    def __init__(self, capacity, refill_per_sec):
        self.capacity = capacity          # most tickets the bucket can hold
        self.rate = refill_per_sec        # how fast tickets come back
        self.tokens = capacity            # start full
        self.last = 0.0                   # last time we checked (seconds)

    def allow(self, now):
        # add back tickets for the time that has passed, but never above capacity
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1:              # enough for one request?
            self.tokens -= 1
            return True
        return False                      # bucket empty -> "429 Too Many Requests"

bucket = TokenBucket(capacity=3, refill_per_sec=1)
for t in [0, 0, 0, 0, 2]:                 # 4 requests at second 0, one at second 2
    print(f"t={t}s allowed={bucket.allow(t)}")
# t=0s allowed=True
# t=0s allowed=True
# t=0s allowed=True
# t=0s allowed=False
# t=2s allowed=True
```

**What happened:** the first 3 requests used up the bucket, the 4th was refused, and after 2 seconds tickets had refilled so the next one was allowed.

## Watch Out
- Counting only requests isn't enough. One huge prompt can cost more than 100 small ones, so also limit **tokens or spending** per user.
- When the provider says 429, don't retry instantly in a loop. Wait, and wait longer each time.

## Remember
- Rate limiting protects your **budget, servers and fairness**.
- The token bucket allows short bursts but caps the average.
- Limit by requests **and** by tokens or cost.

## Related Questions
- [What is the role of an API gateway?](02-What-is-the-role-of-an-API-gateway.md)
- [How do you manage API costs and rate limits?](../Week-06-Working-with-LLMs-and-Model-APIs/17-How-do-you-manage-API-costs-and-rate-limits.md)
- [What is token consumption?](11-What-is-token-consumption.md)
- [How do you handle API errors?](../Week-05-Python-for-LLM-Engineering/15-How-do-you-handle-API-errors.md)
