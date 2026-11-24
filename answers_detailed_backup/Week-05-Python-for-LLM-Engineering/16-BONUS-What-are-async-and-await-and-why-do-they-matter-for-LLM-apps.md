# What are async and await, and why do they matter for LLM apps?

> Week 05 – Python for LLM Engineering · Question 16 · **BONUS**
>
> Bonus question (added)

## Short Answer
`async def` defines a **coroutine**: a function that can pause while it waits for I/O. `await` marks those pause points, letting Python's **event loop** run other work in the meantime, all on a single thread. LLM apps spend most of their time *waiting* on network responses that take seconds, so `asyncio` lets one process run many LLM calls concurrently (for example, 50 summaries at once instead of one after another) and keeps web servers responsive while they stream tokens.

## Detailed Explanation

**Sync vs async for three 2-second LLM calls:**
```
Sync :  [call A ██████][call B ██████][call C ██████]   ≈ 6 s
Async:  [call A ██████]
        [call B ██████]                                 ≈ 2 s
        [call C ██████]
```

| Concept | Meaning |
|---|---|
| `async def f()` | Defines a coroutine function. Calling it returns a coroutine object; it doesn't run yet. |
| `await x` | Pause here until `x` finishes; other tasks may run meanwhile. Only valid inside `async def`. |
| `asyncio.run(main())` | Starts the event loop (entry point of a script). |
| `asyncio.gather(*coros)` | Run several coroutines concurrently and collect their results in order. |
| `asyncio.Semaphore(n)` | Limit concurrency, which is essential to respect rate limits. |
| `asyncio.TaskGroup` (3.11+) | Structured concurrency: if one task fails, the others are cancelled. |

**Concurrency ≠ parallelism:** async interleaves *waiting*. It doesn't speed up CPU-heavy work such as local model inference on CPU or big NumPy jobs. Use processes or a GPU server for that.

**Where it matters in LLM engineering:**
- Batch processing (classify or summarise thousands of documents).
- Calling several tools or retrievers at once in an agent or RAG pipeline.
- Web backends (FastAPI) serving many users while streaming responses.

Async-capable clients include `anthropic.AsyncAnthropic`, `httpx.AsyncClient` and `ollama.AsyncClient`.

## Example
**Analogy:** a sync chef boils pasta and stares at the pot until it's done before starting the sauce. An async chef starts the pasta, makes the sauce while it boils, and checks the oven too. Same single chef, far less idle time.

```python
import asyncio
import time
import anthropic

client = anthropic.AsyncAnthropic()      # reads ANTHROPIC_API_KEY
limit = asyncio.Semaphore(5)             # at most 5 requests in flight

async def summarize(text: str) -> str:
    async with limit:
        resp = await client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=60,
            system="Summarise in at most 8 words.",
            messages=[{"role": "user", "content": text}],
        )
        return resp.content[0].text

async def main():
    docs = ["Python is popular for AI because of its libraries.",
            "HTTP 429 means you are being rate limited.",
            "NumPy arrays enable fast vectorised maths."]
    start = time.perf_counter()
    summaries = await asyncio.gather(*(summarize(d) for d in docs))
    for s in summaries:
        print("-", s)
    print(f"done in {time.perf_counter() - start:.1f}s")

asyncio.run(main())
# - Python dominates AI thanks to libraries.
# - 429 signals rate limiting.
# - NumPy enables fast vectorised maths.
# done in 1.2s   (roughly one call's latency, not three; output varies)
```

In Jupyter an event loop is already running, so use `await main()` instead of `asyncio.run(main())`.

## Common Mistakes / Misconceptions
- **Calling blocking code inside `async def`** (`time.sleep`, `requests.get`, the sync `anthropic.Anthropic` client). It freezes the whole event loop. Use `await asyncio.sleep`, `httpx.AsyncClient` and `AsyncAnthropic`, or offload with `asyncio.to_thread`.
- **Forgetting `await`.** You get a coroutine object (and a "never awaited" warning) instead of a result.
- **Unlimited concurrency** with `gather` over 10,000 items, which immediately triggers 429s. Use a semaphore.
- **Expecting async to speed up CPU-bound work.** It only helps while waiting on I/O.

## Key Takeaways
- `async`/`await` provide cooperative concurrency on a single thread for I/O-bound work.
- LLM calls are slow network I/O, which is exactly the case async handles best.
- Use `gather`/`TaskGroup` for fan-out and `Semaphore` to stay under rate limits.
- Never block the event loop. Use async-native clients.

## Related Questions
- [How do you consume a REST API using Python?](14-How-do-you-consume-a-REST-API-using-Python.md)
- [How do you handle API errors?](15-How-do-you-handle-API-errors.md)
- [What is response streaming?](../Week-06-Working-with-LLMs-and-Model-APIs/14-What-is-response-streaming.md)
- [How do you manage API costs and rate limits?](../Week-06-Working-with-LLMs-and-Model-APIs/17-How-do-you-manage-API-costs-and-rate-limits.md)
