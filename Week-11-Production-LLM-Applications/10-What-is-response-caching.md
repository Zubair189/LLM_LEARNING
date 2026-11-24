# What is response caching?

> Week 11 – Production LLM Applications · Question 10

## Short Answer
Response caching stores the result of an LLM call so that an identical (or very similar) request can be answered from the cache instead of calling the model again. That saves money and cuts latency from seconds to milliseconds. **Exact caching** keys on a hash of the model, prompt and parameters. **Semantic caching** matches new questions to past ones by embedding similarity. Response caching is different from provider-side **prompt caching**, which reuses a processed prompt *prefix* but still generates a fresh answer.

## Detailed Explanation
```
request ─► build cache key = hash(model + system + messages + params)
              │
        ┌─────┴──────┐
     hit│            │miss
        ▼            ▼
  return cached   call LLM ─► store (key → answer, TTL) ─► return
```

**Types of caching in LLM apps**
| Type | Where | What's reused | Fresh generation? |
|---|---|---|---|
| Exact response cache | Your app (Redis/memory) | Full answer for identical input | No |
| Semantic cache | Your app + embeddings | Answer for a *similar* question (cosine ≥ threshold) | No |
| Prompt caching | Provider (`cache_control`) | Processed long prompt prefix (system prompt, docs) | Yes; cheaper input |
| Embedding cache | Your app | Embeddings of chunks/queries | n/a |

**When caching is appropriate**
- Deterministic-ish tasks: FAQs, classification, summaries of static documents.
- Set `temperature=0` for cacheable tasks and include all parameters in the key.
- Use a TTL so answers refresh when underlying data changes.
- Be careful with personalised or permissioned data: include the user or tenant in the key, or don't cache.

## Example
**Analogy:** A help-desk worker keeps a sheet of answers to frequent questions. When the same question arrives they read from the sheet instead of researching it again.

```python
import hashlib, json, time
import anthropic

client = anthropic.Anthropic()
cache: dict[str, tuple[float, str]] = {}   # key -> (expires_at, answer); use Redis in prod
TTL = 3600

def cached_ask(question: str, system: str = "Answer briefly.") -> tuple[str, bool]:
    payload = {"m": "claude-haiku-5-5", "s": system, "q": question.strip().lower(), "t": 0}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    hit = cache.get(key)
    if hit and hit[0] > time.time():
        return hit[1], True
    resp = client.messages.create(model="claude-haiku-5-5", max_tokens=200, temperature=0,
                                  system=system,
                                  messages=[{"role": "user", "content": question}])
    answer = resp.content[0].text
    cache[key] = (time.time() + TTL, answer)
    return answer, False

print(cached_ask("What is a token?")[1])   # -> False (miss, ~1s)
print(cached_ask("what is a token? ")[1])  # -> True  (hit, ~0 ms, no cost)
```

Provider prompt caching for a long, stable system prompt:
```python
resp = client.messages.create(
    model="claude-sonnet-5-5", max_tokens=300,
    system=[{"type": "text", "text": LONG_POLICY_DOC,
             "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": "Can I return an opened item?"}])
```

## Common Mistakes / Misconceptions
- Confusing response caching with prompt caching. Only response caching skips generation.
- Caching answers built from user-specific data and serving them to other users.
- No TTL or invalidation, so stale answers live forever.
- Setting the semantic-cache threshold too loose, so "cancel my order" matches "track my order".

## Key Takeaways
- Caching cuts cost and latency for repeated requests.
- Key on everything that affects the output (model, prompt, params, user/tenant).
- Use TTLs and invalidation; be careful with semantic thresholds.
- Combine response caching (your app) with prompt caching (provider) for long prompts.

## Related Questions
- [How do you reduce LLM API costs?](12-How-do-you-reduce-LLM-API-costs.md)
- [What is token consumption?](11-What-is-token-consumption.md)
- [What is semantic similarity?](../Week-07-Embeddings-and-Vector-Databases/06-What-is-semantic-similarity.md)
- [What is the temperature parameter?](../Week-06-Working-with-LLMs-and-Model-APIs/12-What-is-the-temperature-parameter.md)
