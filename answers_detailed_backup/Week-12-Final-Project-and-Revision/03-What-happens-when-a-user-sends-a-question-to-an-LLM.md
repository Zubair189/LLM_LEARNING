# What happens when a user sends a question to an LLM?

> Week 12 – Final Project and Revision · Question 3

## Short Answer
In a real application the question first passes through your **backend** (auth, rate limiting, input guardrails, caching), which **builds a prompt** from the system prompt, conversation history, and optionally retrieved documents. The provider then **tokenizes** the prompt, runs it through the Transformer to get a probability distribution over the next token, **samples** a token, appends it, and repeats until a stop condition, often **streaming** tokens back. Your backend then applies output checks, logs tokens/latency/cost, and returns the answer.

## Detailed Explanation

### End-to-end request path

```
 User ──"What is our refund policy?"──▶ Client (web/mobile)
                                          │ HTTPS
                                          ▼
 ┌────────────────────────── Your backend (FastAPI) ──────────────────────────┐
 │ 1 Auth + API gateway   2 Rate limit (token bucket)   3 Input guardrails    │
 │ 4 Cache lookup ──hit──▶ return cached answer                               │
 │ 5 Retrieve context (RAG: embed query → vector DB → top-k chunks)           │
 │ 6 Build prompt: system + history (trimmed) + context + question            │
 └──────────────────────────────────┬─────────────────────────────────────────┘
                                    │ messages.create(...)  (API key from env)
                                    ▼
 ┌────────────────────────── LLM provider ────────────────────────────────────┐
 │ 7 Tokenize → IDs   8 Embeddings + positions → N Transformer layers          │
 │ 9 Logits → softmax → sample (temperature/top-p) → next token               │
 │ 10 Append token, repeat (KV cache) until end-of-turn or max_tokens         │
 └──────────────────────────────────┬─────────────────────────────────────────┘
                                    │ streamed tokens + usage
                                    ▼
 ┌──────────────── Backend ───────────────┐
 │ 11 Output guardrails / format check    │
 │ 12 Log latency, tokens, cost, trace id │──▶ observability
 │ 13 Save to history + cache             │
 └──────────────────┬─────────────────────┘
                    ▼
               User sees answer (streaming)
```

### Step highlights
1. **Prompt assembly** — the model is stateless; every call must include the system prompt and any history you want it to "remember".
2. **Tokenization** — text becomes subword tokens; input tokens are billed and count against the context window.
3. **Forward pass** — the whole prompt is processed in parallel ("prefill"); the model outputs logits for the next token.
4. **Decoding** — one token at a time; temperature/top-p control randomness. A KV cache avoids recomputing earlier tokens.
5. **Stop** — when the model emits an end-of-turn token, a stop sequence, or `max_tokens` is reached (`stop_reason` tells you which).
6. **Return** — streamed for better perceived latency (time-to-first-token matters to users).

### Latency breakdown (illustrative)
| Part | Typical driver |
|---|---|
| Network + backend | Usually small (tens of ms) |
| Retrieval | Embedding + vector search (tens–hundreds of ms) |
| Prefill | Grows with prompt length |
| Generation | Grows with number of output tokens — usually the largest part |

## Example
**Analogy:** Ordering at a restaurant. The host checks your reservation (auth), the waiter writes your order with your table's past orders and today's specials (prompt with history + RAG), the kitchen cooks one dish at a time (token-by-token generation), the waiter checks the plate before serving (output guardrails), and the manager records the order (logging).

**Python – streaming endpoint showing the pipeline:**
```python
import anthropic
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()
client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY

class Ask(BaseModel):
    question: str

def retrieve(q: str) -> str:            # stand-in for RAG
    return "Refunds are accepted within 30 days with a receipt."

@app.post("/ask")
async def ask(body: Ask):
    context = retrieve(body.question)
    async def gen():
        async with client.messages.stream(
            model="claude-sonnet-5-5",
            max_tokens=300,
            system="Answer only from the context. If missing, say you don't know.",
            messages=[{"role": "user",
                       "content": f"<context>{context}</context>\n\n{body.question}"}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
    return StreamingResponse(gen(), media_type="text/plain")

# curl -N -X POST localhost:8000/ask -H 'content-type: application/json' \
#      -d '{"question":"What is the refund policy?"}'
# Refunds are accepted within 30 days with a receipt.   (streamed token by token)
```

## Common Mistakes / Misconceptions
- **"The LLM remembers the previous message."** Only if your app re-sends it.
- **"The LLM searches the internet/database by itself."** It only sees what you put in the prompt or what tools you give it.
- **Ignoring output tokens in latency/cost.** Generation is sequential, so long answers are slow and expensive.
- **Skipping backend checks** and calling the provider directly from the browser — this leaks the API key and bypasses rate limits.

## Key Takeaways
- The request flows: client → backend (auth, limits, guardrails, cache, RAG, prompt) → LLM (tokenize, forward pass, sample, repeat) → backend (checks, logs) → user.
- LLMs are stateless; your app supplies memory and knowledge.
- Generation is autoregressive: one token at a time until a stop condition.
- Streaming improves perceived latency; logging tokens enables cost and quality monitoring.

## Related Questions
- [What is inference?](../Week-02-Introduction-to-LLMs-and-Generative-AI/11-What-is-inference.md)
- [What is a token?](../Week-02-Introduction-to-LLMs-and-Generative-AI/06-What-is-a-token.md)
- [What is the temperature parameter?](../Week-06-Working-with-LLMs-and-Model-APIs/12-What-is-the-temperature-parameter.md)
- [What is response streaming?](../Week-06-Working-with-LLMs-and-Model-APIs/14-What-is-response-streaming.md)
- [What is context injection?](../Week-08-Retrieval-Augmented-Generation-RAG/10-What-is-context-injection.md)
- [How do you maintain conversation history?](../Week-11-Production-LLM-Applications/04-How-do-you-maintain-conversation-history.md)
- [What is the role of an API gateway?](../Week-11-Production-LLM-Applications/02-What-is-the-role-of-an-API-gateway.md)
- [Explain Transformer architecture.](02-Explain-Transformer-architecture.md)
