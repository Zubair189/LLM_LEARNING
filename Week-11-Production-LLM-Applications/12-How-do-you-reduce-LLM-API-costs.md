# How do you reduce LLM API costs?

> Week 11 – Production LLM Applications · Question 12

## Short Answer
Reduce costs by sending fewer tokens, generating fewer tokens and using cheaper paths when possible. In practice that means routing simple tasks to a smaller model, trimming prompts, history and RAG context, capping `max_tokens`, caching responses, and using provider prompt caching for long, repeated prefixes. Batching non-urgent work and setting budgets, alerts and rate limits stop surprises. Measure token usage first so you optimise the biggest cost driver.

> **Distinction:** Q11 (token consumption) explains *what* you are paying for and how to measure it. This question is the *playbook for lowering* it.

## Detailed Explanation
| Lever | How | Typical impact (illustrative) |
|---|---|---|
| Model routing | Use `claude-haiku-5-5` for classification, extraction and routing; `claude-sonnet-5-5` only for hard reasoning | Large |
| Response caching | Exact/semantic cache for repeated questions | Large for FAQ-style traffic |
| Prompt caching | `cache_control` on long, stable system prompts or documents | Cheaper repeated input |
| Shorter prompts | Remove verbose instructions and redundant examples | Medium |
| Context control | Summarise history, fewer/smaller RAG chunks, trim tool output | Medium–large |
| Output limits | Lower `max_tokens`, ask for concise or structured output | Medium |
| Batch processing | Use provider batch APIs for offline jobs (often discounted) | Medium |
| Avoid wasted calls | Validate input first, dedupe, stop agent loops after N steps | Medium |
| Budgets & limits | Per-user quotas, spend alerts, rate limiting | Prevents spikes |

```
request ─► cache? ──hit──► $0
             │miss
             ▼
        classify difficulty (cheap model)
         ├─ simple ─► small model, short max_tokens
         └─ complex ─► large model + cached system prompt + top-3 chunks
```

Always re-run your evaluation set after a cost change. A cheaper pipeline that gives wrong answers is not a saving.

## Example
**Analogy:** Running a delivery business: bikes for short trips and trucks only for heavy loads, combined orders (batching), and the same route not driven twice (caching).

```python
import anthropic

client = anthropic.Anthropic()
SYSTEM_DOC = "...long product manual (thousands of tokens)..."

def route(question: str) -> str:
    r = client.messages.create(
        model="claude-haiku-5-5", max_tokens=5, temperature=0,
        system="Reply with exactly SIMPLE or COMPLEX.",
        messages=[{"role": "user", "content": question}])
    return r.content[0].text.strip()

def answer(question: str) -> str:
    model = "claude-haiku-5-5" if route(question) == "SIMPLE" else "claude-sonnet-5-5"
    r = client.messages.create(
        model=model, max_tokens=300,
        system=[{"type": "text", "text": SYSTEM_DOC,
                 "cache_control": {"type": "ephemeral"}}],   # reuse long prefix
        messages=[{"role": "user", "content": question}])
    print(model, r.usage.input_tokens, r.usage.output_tokens)
    return r.content[0].text

answer("What are your opening hours?")
# -> claude-haiku-5-5 1850 22        (illustrative)
answer("Compare warranty options for a business with 40 devices.")
# -> claude-sonnet-5-5 1870 240      (illustrative)
```
Check the provider pricing page for current per-token and cache prices; numbers above are illustrative only.

## Common Mistakes / Misconceptions
- Optimising blindly without measuring which endpoint or user burns the most tokens.
- Using the most powerful model for every call "to be safe".
- Expecting prompt caching to help with prompts that change at the start. Only a stable prefix is cached.
- Cutting context so aggressively that answer quality drops, with no eval to catch it.

## Key Takeaways
- Measure first, then attack the biggest driver.
- Route by difficulty: small model by default, big model when needed.
- Cache responses and long prompt prefixes; trim context and output.
- Guard with budgets, quotas and evals so savings don't break quality.

## Related Questions
- [What is token consumption?](11-What-is-token-consumption.md)
- [What is response caching?](10-What-is-response-caching.md)
- [How do you manage API costs and rate limits?](../Week-06-Working-with-LLMs-and-Model-APIs/17-How-do-you-manage-API-costs-and-rate-limits.md)
- [What is context management?](05-What-is-context-management.md)
