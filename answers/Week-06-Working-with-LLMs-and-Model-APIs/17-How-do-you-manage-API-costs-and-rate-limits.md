# How do you manage API costs and rate limits?

> Week 06 – Working with LLMs and Model APIs · Question 17

## In One Line
You **keep costs down** by sending and receiving fewer tokens on the right-sized model, and you **handle rate limits** by slowing down and retrying politely.

## Simple Explanation
**Costs**

Cloud LLMs charge per **token** (small piece of text):

**cost = input tokens × input price + output tokens × output price**

Output tokens usually cost several times more than input tokens. Each reply tells you how many tokens were used (`resp.usage`).

Ways to save money:
1. **Pick the smallest model that does the job well.** This is often the biggest saving.
2. **Set `max_tokens`** and ask for short answers.
3. **Trim the chat history**, or summarise old messages.
4. **Cache:** reuse answers to repeated questions, and use prompt caching for long, repeated instructions.
5. **Batch:** send non-urgent jobs through a batch API, which is cheaper.
6. **Set spending limits and alerts** in the provider's console.

**Rate limits**

A **rate limit** is a cap on how much you can send per minute (requests and tokens). Go over it and you get error **429 Too Many Requests**.

How to handle it:
1. **Wait and retry**, waiting longer each time (1s, 2s, 4s...). This is **exponential backoff**.
2. **Limit how many calls run at once.**
3. The Anthropic SDK already retries for you (`max_retries`, default 2).

**Link to LLMs:** prices change often, so always check the provider's pricing page.

## Real-Life Example
An LLM API is like a metered taxi. You pay per kilometre (token), so take the direct route (short prompts) and a smaller car when you can. When the dispatcher says "all cars busy" (429), you wait a bit before calling again instead of redialling nonstop.

## Code Example
Comparing the cost of a small and a big model.

```python
# ILLUSTRATIVE prices only (US$ per million tokens) - check the real pricing page
PRICES = {"small-model": (1.00, 5.00), "big-model": (5.00, 25.00)}  # (input, output)

def cost(model, input_tokens, output_tokens):
    price_in, price_out = PRICES[model]
    return input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out

for model in PRICES:                              # same job on both models
    one_call = cost(model, input_tokens=500, output_tokens=200)
    print(f"{model}: ${one_call:.4f} per call, ${one_call * 100_000:.0f} per 100k calls")

# Output:
# small-model: $0.0015 per call, $150 per 100k calls
# big-model: $0.0075 per call, $750 per 100k calls
```

**What happened:** one call costs a fraction of a cent, but at 100,000 calls the bigger model costs five times more. For rate limits, `anthropic.Anthropic(max_retries=5)` adds automatic waiting and retrying.

## Watch Out
- Don't retry instantly in a loop after a 429. That makes it worse. Wait longer each time.
- Don't use the biggest model for everything "just in case".

## Remember
- Cost = **tokens × price**; output tokens cost more.
- Save with **smaller models, shorter prompts, caching and batching**.
- On **429**, back off and retry.

## Related Questions
- [How do you handle API errors?](../Week-05-Python-for-LLM-Engineering/15-How-do-you-handle-API-errors.md)
- [What are async and await, and why do they matter for LLM apps?](../Week-05-Python-for-LLM-Engineering/16-BONUS-What-are-async-and-await-and-why-do-they-matter-for-LLM-apps.md)
- [What is the difference between local and cloud LLMs?](06-What-is-the-difference-between-local-and-cloud-LLMs.md)
- [What is an API key?](08-What-is-an-API-key.md)
