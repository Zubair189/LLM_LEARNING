# What is response caching?

> Week 11 – Production LLM Applications · Question 10

## In One Line
Response caching means **saving a model's answer** so that when the same question comes again, you reuse it instead of paying for a new call.

## Simple Explanation
Many users ask the same things: "What are your opening hours?", "How do I reset my password?". Calling the model every time is slow (seconds) and costs money.

A **cache** is a quick-access store of saved answers. It works like this:

1. A question arrives.
2. Turn it into a **key**, e.g. the question in lowercase with spaces trimmed.
3. Is the key already in the cache? **Yes (a "hit")** → return the saved answer instantly, for free.
4. **No (a "miss")** → call the model, save the answer under that key, then return it.

```
Question → Cache? ──hit──→ saved answer (fast, free)
              └──miss──→ LLM → save answer → return
```

**Two kinds of response cache:**
- **Exact cache:** only matches the same question text.
- **Semantic cache:** also matches questions with the *same meaning* ("opening times?" ≈ "when are you open?") by comparing embeddings.

**Good practice:**
- Give saved answers an **expiry time (TTL)** so they refresh when your data changes.
- Don't share cached answers between users if the answer contains **personal data**.

**Not the same as "prompt caching":** prompt caching is a provider feature that reuses a long, repeated *start* of a prompt to make input cheaper, but the model still writes a fresh answer. Response caching skips the model completely.

## Real-Life Example
A help-desk worker keeps a sheet of answers to common questions. When a familiar question comes in, they read the answer off the sheet instead of researching it again.

## Code Example
A simple dict cache with a 1-hour expiry. The "model" takes 1 second, so you can feel the difference.

```python
import time

cache = {}                                   # question -> (answer, time saved)
TTL = 3600                                   # keep answers for 1 hour

def slow_llm(question):                      # stand-in for a real (slow, paid) model call
    time.sleep(1)
    return "A token is a small piece of text."

def ask(question):
    key = question.strip().lower()           # "What is a token?" and "what is a token? " match
    if key in cache and time.time() - cache[key][1] < TTL:
        return cache[key][0], "cache hit"    # fast and free
    answer = slow_llm(question)              # slow and costs money
    cache[key] = (answer, time.time())       # remember it for next time
    return answer, "cache miss"

print(ask("What is a token?"))     # ('A token is a small piece of text.', 'cache miss')  ~1 s
print(ask("what is a token? "))    # ('A token is a small piece of text.', 'cache hit')   instant
```

**What happened:** the first call waited for the "model". The second, slightly different-looking question matched the same key and came back instantly.

## Watch Out
- Without an expiry time, old answers can stay forever even after facts change.
- A semantic cache that is too loose may match "cancel my order" with "track my order". Test it.

## Remember
- Caching saves **money and time** on repeated questions.
- Build the key from everything that changes the answer.
- Use an **expiry time** and be careful with personal data.

## Related Questions
- [How do you reduce LLM API costs?](12-How-do-you-reduce-LLM-API-costs.md)
- [What is token consumption?](11-What-is-token-consumption.md)
- [What is semantic similarity?](../Week-07-Embeddings-and-Vector-Databases/06-What-is-semantic-similarity.md)
