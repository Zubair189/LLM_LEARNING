# What is an inference API?

> Week 06 – Working with LLMs and Model APIs · Question 7

> Note: The inference API is the **service** you call to get model predictions; an API key ([Q8](08-What-is-an-API-key.md)) is the **credential** that proves you are allowed to call it.

## Short Answer
An inference API is a network endpoint (usually HTTPS + JSON) that runs a trained model on your input and returns its output — you send a prompt, it returns generated text, embeddings or classifications. "Inference" means using a trained model, as opposed to training it. Examples: Anthropic's Messages API (`POST /v1/messages`), OpenAI's API, Hugging Face Inference Endpoints, and Ollama's local `/api/chat`.

## Detailed Explanation
```
  Your app                         Inference API                    GPU cluster
 ─────────── HTTPS POST ───────►  auth, rate limit, queue  ───────► model forward passes
  JSON request                     tokenize input                    generate tokens
 ◄────────── JSON / stream ──────  detokenize, usage stats  ◄────────
```

Anatomy of an LLM inference request:

| Part | Example |
|---|---|
| Endpoint | `https://api.anthropic.com/v1/messages` |
| Auth header | `x-api-key: $ANTHROPIC_API_KEY` |
| Model | `claude-sonnet-5-5` |
| Input | `system` prompt + `messages` list |
| Generation params | `max_tokens`, `temperature`, `stop_sequences` |
| Output | content blocks, `stop_reason`, `usage` (input/output tokens) |

Characteristics:
- **Stateless** – the server does not remember earlier calls; you resend the conversation history each time.
- **Metered** – billed per input and output token (check the provider's pricing page; prices change).
- **Rate-limited** – requests/tokens per minute caps; exceeding them returns HTTP 429.
- **Sync, streaming, or batch** – full response at once, token-by-token streaming, or cheaper asynchronous batch jobs.

SDKs (like the `anthropic` Python package) wrap the raw HTTP call and add types, retries and streaming helpers.

## Example
**Analogy:** An inference API is like a translation agency's phone line: you call, give them text, they return the translation. You never see their staff (the GPUs) — just the service.

Using the SDK:
```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
resp = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=200,
    system="You are a concise assistant.",
    messages=[{"role": "user", "content": "Define inference in one sentence."}],
)
print(resp.content[0].text)
print(resp.stop_reason, resp.usage.input_tokens, resp.usage.output_tokens)
# Expected (values vary): "Inference is using a trained model to make predictions on new input."
# end_turn 22 17
```

The same call as raw HTTP (what the SDK does for you):
```python
import os, requests
r = requests.post(
    "https://api.anthropic.com/v1/messages",
    headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
             "anthropic-version": "2023-06-01",
             "content-type": "application/json"},
    json={"model": "claude-haiku-5-5", "max_tokens": 200,
          "messages": [{"role": "user", "content": "Hi"}]},
    timeout=60,
)
print(r.status_code, r.json()["content"][0]["text"])
# Expected: 200 Hello! How can I help you today?
```

## Common Mistakes / Misconceptions
- **Assuming the API remembers the conversation.** You must send prior messages each call.
- **Confusing inference with training.** Calling the API does not update the model's weights.
- **No timeouts/retries.** Networks fail; use SDK retries or your own backoff.
- **Ignoring `stop_reason`.** `max_tokens` means the answer was cut off.

## Key Takeaways
- Inference API = hosted model behind an HTTP endpoint.
- Requests contain model, prompt/messages and generation parameters; responses include output and token usage.
- Stateless, metered, rate-limited.
- SDKs wrap raw HTTP and add retries and streaming.

## Related Questions
- [What is an API key?](08-What-is-an-API-key.md)
- [What is response streaming?](14-What-is-response-streaming.md)
- [What is an HTTP request?](../Week-05-Python-for-LLM-Engineering/12-What-is-an-HTTP-request.md)
- [How do you manage API costs and rate limits?](17-How-do-you-manage-API-costs-and-rate-limits.md)
