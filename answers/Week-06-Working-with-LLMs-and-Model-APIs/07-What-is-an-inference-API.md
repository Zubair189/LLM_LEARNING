# What is an inference API?

> Week 06 – Working with LLMs and Model APIs · Question 7

> Note: The inference API is the **service** you call, and an API key ([Q8](08-What-is-an-API-key.md)) is the **password-like secret** that lets you call it.

## In One Line
An inference API is an **online service that runs a model for you**: you send it input, it sends back the model's answer.

## Simple Explanation
**Inference** means *using* a trained model to get an answer. (It is the opposite of *training*, which is when the model learns.)

Big models need powerful GPUs. Instead of buying them, you send your request over the internet to a company that runs the model. That door into their service is the **inference API**.

What one call looks like:

1. **You send** an HTTPS request with JSON: which model, your prompt, and settings like `max_tokens` (the longest answer allowed).
2. **The provider** checks your API key, runs the model and generates the answer.
3. **You get back** JSON with the answer text, a `stop_reason` (why it stopped) and `usage` (how many tokens were used).

Things to know:
- **It has no memory.** Each call stands alone, so you resend the chat history every time.
- **You pay per token** you send and receive.
- **There are rate limits**, a cap on requests per minute. Going over returns error 429.

Examples: Anthropic's Messages API, OpenAI's API, Hugging Face Inference Endpoints, and Ollama's local API.

**Link to LLMs:** an SDK like `anthropic` is just a friendly Python wrapper around this API.

## Real-Life Example
An inference API is like a translation agency's phone line. You call, read them your text, and they give you the translation. You never see the translators (the GPUs); you just use the service.

## Code Example
```python
import anthropic                                      # pip install anthropic

client = anthropic.Anthropic()                        # reads ANTHROPIC_API_KEY
resp = client.messages.create(                        # one inference call
    model="claude-sonnet-5-5",                        # which model runs it
    max_tokens=1024,                                  # longest answer allowed
    system="You are a concise assistant.",            # instructions
    messages=[{"role": "user", "content": "Define inference in one sentence."}],
)
print(resp.content[0].text)                           # the answer
print(resp.stop_reason, resp.usage.input_tokens, resp.usage.output_tokens)  # why it stopped + tokens used

# Output (e.g.):
# Inference is using a trained model to make predictions on new input.
# end_turn 22 17
```

**What happened:** our prompt went to Anthropic's servers, the model ran there, and we got back the answer plus the token count used for billing.

## Watch Out
- Calling the API does **not** train or change the model.
- If `stop_reason` is `"max_tokens"`, the answer was cut off. Raise `max_tokens` or ask for a shorter answer.

## Remember
- Inference = **using** a model, not training it.
- An inference API = **model behind a web address**.
- No memory, paid per token, rate-limited.

## Related Questions
- [What is an API key?](08-What-is-an-API-key.md)
- [What is response streaming?](14-What-is-response-streaming.md)
- [What is an HTTP request?](../Week-05-Python-for-LLM-Engineering/12-What-is-an-HTTP-request.md)
- [How do you manage API costs and rate limits?](17-How-do-you-manage-API-costs-and-rate-limits.md)
