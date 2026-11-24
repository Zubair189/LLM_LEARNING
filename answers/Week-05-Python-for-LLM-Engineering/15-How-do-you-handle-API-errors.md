# How do you handle API errors?

> Week 05 – Python for LLM Engineering · Question 15

## In One Line
**Retry** the errors that are temporary, **stop and fix** the errors that are your fault, and always show the user a friendly message.

## Simple Explanation
API calls fail sometimes. The trick is to sort errors into two groups.

**1. Temporary errors: try again.**
- **Timeouts** or dropped connections.
- **429 Too Many Requests:** you are sending too fast (a **rate limit**).
- **5xx errors:** the server is busy or has a problem.

**2. Permanent errors: don't retry, fix the request.**
- **400:** your request is wrong, e.g. a bad setting.
- **401:** your API key is missing or wrong.
- **404:** wrong address or model name.

**How to retry well: exponential backoff.** Wait 1 second, then 2, then 4, doubling each time. Add a little random extra wait (called **jitter**) so many users don't all retry at the same moment. Set a maximum number of tries.

Also:
- Always set a **timeout**.
- Have a **fallback**, like a friendly "please try again" message.
- A 200 reply can still be a problem. If the answer was cut off (`stop_reason == "max_tokens"`), handle that too.

**Link to LLMs:** the Anthropic SDK already retries temporary errors for you (`max_retries`, default 2) and gives clear error types like `anthropic.RateLimitError`.

## Real-Life Example
It's like phoning a busy help line. If the line is engaged, you wait a bit longer each time before calling again. But if you dialled a wrong number, calling again won't help; you need to fix the number.

## Code Example
A fake API that is busy twice and then works.

```python
import time

class BusyError(Exception):                     # our own "temporary problem" error
    pass

calls = 0

def flaky_llm_call():                           # pretend API: busy twice, then works
    global calls
    calls += 1
    if calls < 3:
        raise BusyError("429 Too Many Requests")
    return "Here is your answer!"

def ask_with_retry(max_attempts=4):
    for attempt in range(1, max_attempts + 1):
        try:
            return flaky_llm_call()             # try the call
        except BusyError as e:                  # a temporary error: worth retrying
            if attempt == max_attempts:         # out of tries: give up nicely
                return "Sorry, please try again later."
            wait = 2 ** (attempt - 1)           # 1s, 2s, 4s ... (exponential backoff)
            print(f"attempt {attempt} failed ({e}); waiting {wait}s")
            time.sleep(wait)                    # pause before trying again

print(ask_with_retry())

# Output:
# attempt 1 failed (429 Too Many Requests); waiting 1s
# attempt 2 failed (429 Too Many Requests); waiting 2s
# Here is your answer!
```

**What happened:** the first two tries failed, so the code waited longer each time and the third try worked. With the real SDK you can simply write `anthropic.Anthropic(max_retries=5)`.

## Watch Out
- Don't retry everything. Retrying a 401 (bad key) just wastes time.
- Never retry in an endless, instant loop. That makes rate limiting worse.

## Remember
- Temporary (timeout, 429, 5xx) → **retry with backoff**.
- Your fault (400, 401, 404) → **fix, don't retry**.
- Always have a **timeout** and a **friendly fallback**.

## Related Questions
- [How do Python exceptions work?](08-How-do-Python-exceptions-work.md)
- [How do you consume a REST API using Python?](14-How-do-you-consume-a-REST-API-using-Python.md)
- [How do you manage API costs and rate limits?](../Week-06-Working-with-LLMs-and-Model-APIs/17-How-do-you-manage-API-costs-and-rate-limits.md)
