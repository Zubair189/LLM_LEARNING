# How do you monitor latency and failures?

> Week 11 – Production LLM Applications · Question 16

## In One Line
You **time every request and record every error**, watch the slowest requests (not just the average), and have a backup plan when the model fails.

## Simple Explanation
**Latency** means how long a request takes. **Failures** are requests that error out or give a broken answer.

**What to measure:**
1. **p95 latency:** sort all request times; p95 is the time that 95% of requests beat. It shows how slow it is for your unluckiest users. Averages hide those slow ones.
2. **Time to first token:** for streaming chat, how long until the first word appears. This is what users *feel*.
3. **Time per step:** retrieval, model call, tools. This finds the bottleneck.
4. **Errors by type:** timeouts, 429 (rate limited), 5xx (provider problems), and **bad outputs** (e.g. broken JSON).

**What to do when things fail:**
- **Timeout:** don't wait forever; give up after, say, 30 seconds.
- **Retry with backoff:** try again, waiting a bit longer each time (1 s, 2 s, 4 s).
- **Fallback:** if the main model keeps failing, use a backup model or a cached answer.
- **Friendly error:** if all else fails, tell the user nicely.

```
Call LLM → OK? → answer (record time)
   └─ fail → retry (wait longer each time) → still failing? → backup model → still failing? → friendly error
```

**Alerts:** e.g. "p95 above 8 seconds for 5 minutes" or "error rate above 2%".

## Real-Life Example
A delivery company doesn't just track the average delivery time; it watches the slowest 5% of deliveries. When a road is blocked (the provider is down), drivers take a backup route instead of waiting forever.

## Code Example
Retry with backoff, then fall back to a backup model. Then compare average with p95.

```python
import time

def call_model(model, question):                 # pretend the main model is down
    if model == "main-model":
        raise TimeoutError("no reply in 30s")
    return f"answer from {model}"

def ask(question):
    for model in ["main-model", "backup-model"]:  # try main first, then the backup
        for attempt in range(3):                  # up to 3 tries per model
            try:
                return call_model(model, question)
            except TimeoutError:
                wait = 0.1 * 2 ** attempt         # back off: 0.1s, 0.2s, 0.4s
                print(f"{model} failed (try {attempt + 1}), waiting {wait}s")
                time.sleep(wait)
    return "Sorry, please try again later."      # friendly message if everything fails

print(ask("Hi"))
latencies = sorted([150] * 18 + [1800, 2500])     # 20 requests, in milliseconds
print("average:", sum(latencies) / len(latencies), "ms")
print("p95:", latencies[int(0.95 * (len(latencies) - 1))], "ms")
# main-model failed (try 1), waiting 0.1s
# main-model failed (try 2), waiting 0.2s
# main-model failed (try 3), waiting 0.4s
# answer from backup-model
# average: 350.0 ms
# p95: 1800 ms
```

**What happened:** the user still got an answer thanks to the backup. And the average (350 ms) looked fine, while p95 (1,800 ms) revealed that some users waited much longer.

## Watch Out
- Don't retry errors that will never succeed, like a bad request (400) or a wrong API key.
- "HTTP 200 OK" doesn't always mean success. An empty or broken answer is also a failure.

## Remember
- Watch **p95** and time to first token, not just the average.
- Use **timeouts, retries with backoff and fallbacks**.
- Count bad answers as failures too.

## Related Questions
- [What is observability?](15-What-is-observability.md)
- [How do you handle API errors?](../Week-05-Python-for-LLM-Engineering/15-How-do-you-handle-API-errors.md)
- [What is response streaming?](../Week-06-Working-with-LLMs-and-Model-APIs/14-What-is-response-streaming.md)
- [What is rate limiting?](09-What-is-rate-limiting.md)
