# What are async and await, and why do they matter for LLM apps?

> Week 05 – Python for LLM Engineering · Question 16 · **BONUS**
>
> Bonus question (added)

## In One Line
`async` and `await` let Python **start many slow tasks and wait for them at the same time**, instead of one after another.

## Simple Explanation
An LLM call is slow mostly because your program is **waiting** for the answer to come back over the internet. While it waits, the computer is doing nothing.

`async`/`await` fixes that:

1. **`async def`** makes a function that is allowed to **pause** while it waits.
2. **`await`** marks the spot where it pauses. While paused, Python runs other tasks.
3. The **event loop** is the manager that switches between paused tasks. `asyncio.run(main())` starts it.
4. **`asyncio.gather(...)`** starts several tasks together and collects all their results.

So three 1-second calls take about **1 second** in total, not 3.

Async only helps with **waiting** (network, files). It does not make heavy maths faster.

**Link to LLMs:** use it to summarise many documents at once, or to serve many chat users from one web server. The SDK has an async client: `anthropic.AsyncAnthropic()`. Use `asyncio.Semaphore` to limit how many calls run at once, so you don't hit rate limits.

## Real-Life Example
A normal chef puts pasta on to boil and stares at the pot until it's done before starting the sauce. An async chef starts the pasta, makes the sauce while it boils, and checks the oven too. Same one chef, much less standing around.

## Code Example
We fake the LLM call with a 1-second sleep.

```python
import asyncio
import time

async def fake_llm(question):                 # "async def" = can pause while waiting
    await asyncio.sleep(1)                    # pretend the API takes 1 second
    return f"answer to {question}"

async def main():
    start = time.perf_counter()
    answers = await asyncio.gather(           # start all 3 at the same time
        fake_llm("Q1"), fake_llm("Q2"), fake_llm("Q3")
    )
    print(answers)
    print(f"took {time.perf_counter() - start:.1f}s")  # about 1s, not 3s

asyncio.run(main())                           # start the event loop

# Output:
# ['answer to Q1', 'answer to Q2', 'answer to Q3']
# took 1.0s
```

**What happened:** all three "calls" waited at the same time, so the total was about 1 second. With a real API you would `await client.messages.create(...)` using `AsyncAnthropic`.

## Watch Out
- Don't use blocking code like `time.sleep()` or `requests.get()` inside `async def`. It freezes everything. Use `await asyncio.sleep()` and async clients.
- Forgetting `await` gives you a "coroutine object" instead of the result.

## Remember
- `async def` = can pause. `await` = pause here.
- Great for **many slow network calls** at once.
- Limit how many run together to respect **rate limits**.

## Related Questions
- [How do you consume a REST API using Python?](14-How-do-you-consume-a-REST-API-using-Python.md)
- [How do you handle API errors?](15-How-do-you-handle-API-errors.md)
- [What is response streaming?](../Week-06-Working-with-LLMs-and-Model-APIs/14-What-is-response-streaming.md)
- [How do you manage API costs and rate limits?](../Week-06-Working-with-LLMs-and-Model-APIs/17-How-do-you-manage-API-costs-and-rate-limits.md)
