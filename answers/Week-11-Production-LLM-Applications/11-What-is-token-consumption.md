# What is token consumption?

> Week 11 – Production LLM Applications · Question 11

## In One Line
Token consumption is **how many tokens a request uses**, counting both what you send in and what the model writes back, and it decides most of your cost.

## Simple Explanation
A **token** is a small piece of text, roughly ¾ of an English word. LLM providers charge **per token**.

Every call uses two kinds:
1. **Input tokens:** everything you send. This is more than just the user's question:
   - the system prompt,
   - tool descriptions,
   - the conversation history (grows every turn!),
   - retrieved documents (RAG),
   - and finally the question itself.
2. **Output tokens:** the answer the model writes. These usually cost **more** per token than input, and they are generated one by one, so long answers are also **slower**.

**Why measure it?**
- **Cost:** roughly `input tokens × input price + output tokens × output price`.
- **Speed:** more output tokens = longer wait.
- **Limits:** input + output must fit in the context window.

**How to measure it:** every API response includes a `usage` section, e.g. `resp.usage.input_tokens` and `resp.usage.output_tokens`. Log these on every call, then add them up per user and per feature.

This question is about **measuring** where tokens go. Q12 is about **reducing** the cost.

## Real-Life Example
A taxi meter. You pay for every kilometre, and long detours quietly push up the fare. A bloated chat history is like a detour you didn't notice: the meter keeps running.

## Code Example
Turn the `usage` numbers into a cost. The prices are **illustrative only**; always check the provider's pricing page.

```python
# Numbers like these come back in resp.usage after every API call
usage = {"input_tokens": 1200, "output_tokens": 300}

# ILLUSTRATIVE prices in dollars per 1 million tokens (check the real pricing page)
PRICE_IN = 3.00
PRICE_OUT = 15.00

cost_in = usage["input_tokens"] / 1_000_000 * PRICE_IN     # cost of what we sent
cost_out = usage["output_tokens"] / 1_000_000 * PRICE_OUT  # cost of what we got back
print(f"input:  ${cost_in:.4f}")
print(f"output: ${cost_out:.4f}")
print(f"total:  ${cost_in + cost_out:.4f}")
print(f"10,000 calls a day: ${(cost_in + cost_out) * 10_000:.2f}")
# input:  $0.0036
# output: $0.0045
# total:  $0.0081
# 10,000 calls a day: $81.00
```

**What happened:** 300 output tokens cost more than 1,200 input tokens, because output is priced higher. Tiny costs per call add up fast at scale.

## Watch Out
- Don't count only the user's message. The system prompt, history and documents are often the **biggest** part.
- Words are not tokens. Code, numbers and non-English text often use more tokens per word.

## Remember
- Tokens = **input + output**; both are billed.
- History, RAG context and tool descriptions are the **hidden** token users.
- **Log `usage` on every call**, then optimise.

## Related Questions
- [How do you reduce LLM API costs?](12-How-do-you-reduce-LLM-API-costs.md)
- [What is a token?](../Week-02-Introduction-to-LLMs-and-Generative-AI/06-What-is-a-token.md)
- [How does tokenization work?](../Week-02-Introduction-to-LLMs-and-Generative-AI/07-How-does-tokenization-work.md)
- [What is context management?](05-What-is-context-management.md)
