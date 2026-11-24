# What is token consumption?

> Week 11 – Production LLM Applications · Question 11

## Short Answer
Token consumption is the number of tokens an LLM request uses: **input tokens** (system prompt, history, retrieved context, tool definitions, the question) plus **output tokens** (the generated answer). It is the main driver of cost, latency and context-window usage. Providers bill per token, usually charging more for output than input. Measuring it per request, user and feature is the foundation of cost control.

> **Distinction:** this question is about *measuring and understanding* where tokens go. Q12 is about *strategies to reduce* the resulting cost.

## Detailed Explanation
**Where input tokens hide**
```
Total input = system prompt
            + tool/function definitions
            + conversation history (grows every turn!)
            + retrieved RAG chunks
            + current user message
Total output = generated answer (+ any reasoning/tool-call text)
```

**Why it matters**
| Aspect | Effect of more tokens |
|---|---|
| Cost | Roughly linear: `in_tokens × in_price + out_tokens × out_price` |
| Latency | Output tokens are generated one by one, so long answers are slow |
| Context limit | Input + output must fit the window |
| Throughput | Provider limits are often in tokens per minute |

**Measuring it**
- The API response includes `usage.input_tokens` and `usage.output_tokens`. Log them on every call.
- Count before sending with the provider's token-counting endpoint to enforce budgets.
- Aggregate by user, endpoint, model and prompt version on dashboards.

Rule of thumb: in English, 1 token ≈ ¾ of a word. It varies by language and model tokenizer.

## Example
**Analogy:** A taxi meter. You pay for the distance to the destination (input) and back (output), and the return leg costs more per km. Long detours (bloated history) quietly inflate the fare.

```python
import anthropic

client = anthropic.Anthropic()
# ILLUSTRATIVE prices per million tokens – check the provider pricing page for real values
PRICE_IN, PRICE_OUT = 3.00, 15.00

resp = client.messages.create(
    model="claude-sonnet-5-5", max_tokens=300,
    system="You are a concise assistant.",
    messages=[{"role": "user", "content": "Explain embeddings in two sentences."}],
)
u = resp.usage
cost = u.input_tokens / 1e6 * PRICE_IN + u.output_tokens / 1e6 * PRICE_OUT
print(f"in={u.input_tokens} out={u.output_tokens} cost=${cost:.6f}")
# -> in=24 out=61 cost=$0.000987   (illustrative)

# Pre-flight count to enforce a budget:
n = client.messages.count_tokens(model="claude-sonnet-5-5",
        messages=[{"role": "user", "content": "Explain embeddings in two sentences."}])
print(n.input_tokens)   # -> ~16 (illustrative)
```

## Common Mistakes / Misconceptions
- Counting only the user's message and forgetting the system prompt, tool schemas and history.
- Assuming words = tokens. Code, numbers and non-English text often use more tokens.
- Ignoring output tokens. They are usually priced higher and dominate latency.
- Not logging `usage`, so you can't explain a cost spike.

## Key Takeaways
- Tokens = input + output; both are billed, output typically at a higher rate.
- History, RAG context and tool definitions are the hidden token consumers.
- Log `usage` on every call and aggregate by user and feature.
- Measure first (this question), then optimise (Q12).

## Related Questions
- [How do you reduce LLM API costs?](12-How-do-you-reduce-LLM-API-costs.md)
- [What is a token?](../Week-02-Introduction-to-LLMs-and-Generative-AI/06-What-is-a-token.md)
- [How does tokenization work?](../Week-02-Introduction-to-LLMs-and-Generative-AI/07-How-does-tokenization-work.md)
- [What is context management?](05-What-is-context-management.md)
