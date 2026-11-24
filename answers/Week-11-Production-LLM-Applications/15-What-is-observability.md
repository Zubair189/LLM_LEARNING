# What is observability?

> Week 11 – Production LLM Applications · Question 15

## In One Line
Observability means your app **records enough information about what it does** that you can later understand why something went wrong, slow or expensive.

## Simple Explanation
When a user says "the bot gave me a weird answer yesterday", you need to be able to look back and see what happened. Observability is built from three kinds of records:

1. **Logs:** written notes about events, e.g. "called the model, took 1.2 s, used 800 tokens". Best written as **JSON** (structured data) so you can search and count them.
2. **Metrics:** numbers tracked over time, e.g. average speed, error rate, tokens per minute, money per day.
3. **Traces:** the full journey of **one request** through every step, with timings.

For LLM apps you also record: which prompt version was used, which documents were retrieved, which tools were called, token counts and cost.

A trace of one request might look like:

```
Request abc123 (1.9 s total)
 ├─ check user        5 ms
 ├─ search documents  45 ms
 ├─ LLM call          1,780 ms   (2,100 tokens in, 180 out)
 └─ check answer      50 ms
```

Now you can see the LLM call was the slow part.

**Monitoring vs observability:** monitoring tells you *that* something is wrong (an alert fires). Observability gives you the data to find out *why*.

A popular free standard for this is **OpenTelemetry**.

## Real-Life Example
A car has a dashboard (metrics), a trip log (logs), and a GPS replay of the exact route (trace). Together they let the mechanic understand what happened on a bad journey.

## Code Example
Write one structured log line per model call, with a request ID and timing.

```python
import json, time, uuid

def fake_llm(question):                       # stand-in for the real model call
    return "Observability means seeing inside your app.", 12, 9   # answer, tokens in, tokens out

def ask(question):
    request_id = str(uuid.uuid4())[:8]        # one ID to follow this request everywhere
    start = time.perf_counter()               # start a timer
    answer, tokens_in, tokens_out = fake_llm(question)
    ms = round((time.perf_counter() - start) * 1000)
    # one structured (JSON) log line per call: easy to search and chart later
    print(json.dumps({"request_id": request_id, "event": "llm_call", "ms": ms,
                      "tokens_in": tokens_in, "tokens_out": tokens_out}))
    return answer

ask("What is observability?")
# e.g. {"request_id": "7807c1bb", "event": "llm_call", "ms": 0, "tokens_in": 12, "tokens_out": 9}
# (the ID is random and "ms" will be larger with a real model)
```

**What happened:** each call leaves one tidy, searchable record. With a real model you would take the token numbers from `resp.usage`.

## Watch Out
- Prompts and answers can contain **personal data**. Hide (redact) sensitive parts and don't keep logs forever.
- Plain `print("done")` messages are hard to search. Use structured JSON with a request ID.

## Remember
- Observability = **logs + metrics + traces**, plus LLM details like tokens and sources.
- Traces show **where time and tokens go** in one request.
- Protect private data in your logs.

## Related Questions
- [How do you monitor latency and failures?](16-How-do-you-monitor-latency-and-failures.md)
- [What is token consumption?](11-What-is-token-consumption.md)
- [What is LLM evaluation?](13-What-is-LLM-evaluation.md)
- [How do you deploy an LLM application in production?](../Week-12-Final-Project-and-Revision/12-How-do-you-deploy-an-LLM-application-in-production.md)
