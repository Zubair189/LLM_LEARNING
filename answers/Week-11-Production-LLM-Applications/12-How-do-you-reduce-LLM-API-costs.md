# How do you reduce LLM API costs?

> Week 11 – Production LLM Applications · Question 12

## In One Line
You cut LLM costs by **sending fewer tokens, getting fewer tokens back, and using cheaper models** whenever they are good enough.

## Simple Explanation
First **measure** (Q11 shows how) so you know where the money actually goes. Then use these levers, roughly from biggest win to smallest:

1. **Pick the right model for the job.** Send easy tasks (sorting, short answers) to a small, cheap model like `claude-haiku-5-5`. Use a bigger model like `claude-sonnet-5-5` only for hard reasoning. This is called **model routing**.
2. **Cache answers.** If the same question comes again, reuse the saved answer (see Q10).
3. **Use prompt caching.** If every call starts with the same long text (like a big manual), the provider can reuse it so that part is cheaper.
4. **Send less.** Shorten the system prompt, trim old chat history, and send only the top few document chunks.
5. **Ask for less.** Set a sensible `max_tokens` and ask for short answers.
6. **Batch non-urgent jobs.** Many providers give a discount for jobs that can wait.
7. **Set limits.** Per-user quotas, spending alerts and rate limits stop surprise bills.

```
Request → Cache hit? → free
            └─ no → Easy question? → small model
                      └─ hard → big model (+ cached long prompt, few chunks)
```

**Always re-test quality after a cost change.** A cheaper app that gives wrong answers is not a saving.

## Real-Life Example
A delivery company uses bikes for small parcels and trucks only for heavy loads. It combines orders (batching) and doesn't drive the same route twice (caching).

## Code Example
A simple routing rule and the cost difference. The prices are **illustrative, not real**.

```python
# ILLUSTRATIVE prices per 1 million tokens (input, output) - not real prices
PRICES = {"small-model": (1.00, 5.00), "big-model": (3.00, 15.00)}

def pick_model(question):
    hard_words = ["compare", "explain why", "analyse", "plan"]   # signs of a hard task
    if len(question) < 80 and not any(w in question.lower() for w in hard_words):
        return "small-model"              # short and simple -> cheap model
    return "big-model"                    # otherwise -> strong model

def cost(model, tokens_in, tokens_out):
    p_in, p_out = PRICES[model]
    return tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out

for q in ["What are your opening hours?", "Compare your three warranty plans"]:
    m = pick_model(q)
    print(f"{m}: ${cost(m, 500, 200):.4f}")
print(f"Opening hours on big-model would cost: ${cost('big-model', 500, 200):.4f}")
# small-model: $0.0015
# big-model: $0.0045
# Opening hours on big-model would cost: $0.0045
```

**What happened:** the easy question went to the small model and cost a third as much. Real apps often let a small model decide "easy or hard" instead of a word list.

## Watch Out
- Using the most powerful model for everything "to be safe" is the most common money leak.
- Prompt caching only helps when the **start** of the prompt stays the same on every call.

## Remember
- **Measure first**, then fix the biggest cost.
- Small model by default, big model when needed.
- Cache, trim context, limit output, and re-check quality.

## Related Questions
- [What is token consumption?](11-What-is-token-consumption.md)
- [What is response caching?](10-What-is-response-caching.md)
- [How do you manage API costs and rate limits?](../Week-06-Working-with-LLMs-and-Model-APIs/17-How-do-you-manage-API-costs-and-rate-limits.md)
- [What is context management?](05-What-is-context-management.md)
