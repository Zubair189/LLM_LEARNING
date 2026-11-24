# How do you manage API costs and rate limits?

> Week 06 – Working with LLMs and Model APIs · Question 17

## Short Answer
LLM APIs bill per input and output token and cap how many requests and tokens you can send per minute. You manage **cost** by measuring token usage, choosing the smallest model that meets quality, trimming prompts and outputs (`max_tokens`), caching, and using batch APIs for non-urgent work. You manage **rate limits** by handling HTTP 429 errors with exponential backoff and retries, limiting concurrency, and queuing requests — while setting spend alerts in the provider console.

## Detailed Explanation
**Cost formula (always check the provider's pricing page — prices change):**
```
 cost = input_tokens × input_price + output_tokens × output_price
```
Output tokens usually cost several times more than input tokens.

Cost levers:

| Lever | Effect |
|---|---|
| Right-size the model (Haiku → Sonnet → Opus) | Often the biggest saving |
| Cap `max_tokens`, ask for concise output | Cuts expensive output tokens |
| Trim history / context, summarise old turns | Fewer input tokens |
| Prompt caching for long, repeated prefixes (system prompt, docs) | Cheaper, faster repeated input |
| Response caching (same question → stored answer) | Zero cost on repeats |
| Message Batches API for offline jobs | Discounted, asynchronous |
| Count tokens before sending (`client.messages.count_tokens`) | Budget enforcement |

**Rate limits** are typically measured as requests per minute (RPM), input tokens per minute and output tokens per minute, and rise with usage tier. Exceeding them returns **429** (often with a `retry-after` header); **529/overloaded** means the provider is busy.

```
 request ─► 429? ─ yes ─► wait (1s, 2s, 4s, ... + jitter) ─► retry (max N)
              └─ no ─► success
```

The Anthropic SDK already retries 429/5xx/connection errors with backoff (`max_retries`, default 2). Add your own layer for concurrency control (e.g. an `asyncio.Semaphore`) and logging.

## Example
**Analogy:** An LLM API is like a metered taxi with a dispatcher: you pay per kilometre (tokens), so take the direct route (short prompts) and a smaller car when you can; and when the dispatcher says "all cars busy" (429), you wait a bit before calling again instead of redialling nonstop.

```python
import time, random
import anthropic

client = anthropic.Anthropic(max_retries=3)   # SDK-level retries with backoff

# Illustrative prices only — check the provider's pricing page for current values
PRICE_PER_MTOK = {"claude-haiku-5-5": (1.00, 5.00)}  # (input, output) USD per million tokens

def ask(prompt: str, model: str = "claude-haiku-5-5", attempts: int = 4) -> str:
    for i in range(attempts):
        try:
            resp = client.messages.create(model=model, max_tokens=300,
                messages=[{"role": "user", "content": prompt}])
            inp, out = resp.usage.input_tokens, resp.usage.output_tokens
            pin, pout = PRICE_PER_MTOK[model]
            cost = inp / 1e6 * pin + out / 1e6 * pout
            print(f"tokens in={inp} out={out} est_cost=${cost:.6f}")
            return resp.content[0].text
        except anthropic.RateLimitError:
            wait = 2 ** i + random.random()          # exponential backoff + jitter
            print(f"429 rate limited, sleeping {wait:.1f}s")
            time.sleep(wait)
        except anthropic.APIConnectionError:
            time.sleep(2 ** i)
    raise RuntimeError("Gave up after retries")

print(ask("Give one tip to reduce LLM costs."))
# Expected (numbers vary): tokens in=15 out=40 est_cost=$0.000215
# then a one-sentence tip
```

Simple response cache:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_ask(prompt: str) -> str:
    return ask(prompt)   # identical prompts are answered from memory, cost = 0
```

## Common Mistakes / Misconceptions
- **Hardcoding prices** in long-lived code or docs — they change; load from config and re-check.
- **Retrying instantly in a tight loop**, which worsens rate limiting; use backoff + jitter and respect `retry-after`.
- **Defaulting to the biggest model** for every request.
- **Not tracking usage per feature/user**, so a cost spike can't be traced.
- **Retrying non-retryable errors** (400 bad request, 401 auth) — fix them instead.

## Key Takeaways
- Cost = tokens × price; output tokens are pricier — measure via `resp.usage`.
- Right-size models, trim context, cap `max_tokens`, cache, batch.
- Handle 429s with exponential backoff, jitter and limited concurrency.
- Set spend limits/alerts and monitor usage per feature.

## Related Questions
- [How do you handle API errors?](../Week-05-Python-for-LLM-Engineering/15-How-do-you-handle-API-errors.md)
- [What are async and await, and why do they matter for LLM apps?](../Week-05-Python-for-LLM-Engineering/16-BONUS-What-are-async-and-await-and-why-do-they-matter-for-LLM-apps.md)
- [What is the difference between local and cloud LLMs?](06-What-is-the-difference-between-local-and-cloud-LLMs.md)
- [What is an API key?](08-What-is-an-API-key.md)
