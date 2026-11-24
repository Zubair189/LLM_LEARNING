# How do you consume a REST API using Python?

> Week 05 – Python for LLM Engineering · Question 14

## In One Line
You send HTTP requests to the API's web addresses from Python, check that they worked, and turn the JSON reply into Python data.

## Simple Explanation
A **REST API** is a service that lets programs work with it through normal web addresses (URLs) and HTTP methods like GET and POST. "Consuming" it just means **using** it from your code.

The usual recipe:

1. **Read the docs** to learn the URL, method and what data to send.
2. **Load your API key** from an environment variable, never from the code itself.
3. **Send the request** with a library such as `requests` or `httpx`, and set a **timeout**.
4. **Check the status code** (200 means success).
5. **Parse the JSON** reply into a Python dict.
6. **Handle errors**, for example retry if the server is busy.

**Even easier: use an SDK.** An SDK (software development kit) is an official Python library from the provider. It does all the steps above for you: keys, retries and error types. For Claude, that is the `anthropic` package.

**Link to LLMs:** every cloud LLM is a REST API. The SDK is just a friendly wrapper around it.

## Real-Life Example
Using a REST API is like ordering food online. You read the menu (docs), fill in the order form (request), show your membership card (API key), check it says "order accepted" (status code), and then unpack the delivery (parse the JSON).

## Code Example
Using the official SDK, which is the recommended way.

```python
import anthropic                                  # pip install anthropic

client = anthropic.Anthropic(timeout=60.0)        # reads ANTHROPIC_API_KEY; 60s timeout

resp = client.messages.create(                    # sends a POST request for us
    model="claude-sonnet-5-5",                    # which model
    max_tokens=1024,                              # longest reply allowed
    system="Answer in one sentence.",             # instructions for the model
    messages=[{"role": "user", "content": "What is a REST API?"}],
)
print(resp.content[0].text)                       # the answer text
print(resp.usage.input_tokens, resp.usage.output_tokens)  # how many tokens were used

# Output (e.g.):
# A REST API lets programs work with data at web addresses using HTTP methods like GET and POST.
# 25 24
```

**What happened:** the SDK built the HTTP request, added the key, sent it, checked the status and gave us a neat Python object. Without an SDK you would do the same with `httpx.post(...)` and `resp.json()`.

## Watch Out
- Never write your API key directly in the code. Anyone who sees the file can use (and spend) it.
- Always set a timeout and check the status before reading the reply.

## Remember
- Request → check status → parse JSON → handle errors.
- Keep keys in **environment variables**.
- Prefer the **official SDK** when one exists.

## Related Questions
- [What is an HTTP request?](12-What-is-an-HTTP-request.md)
- [How do you handle API errors?](15-How-do-you-handle-API-errors.md)
- [What is an inference API?](../Week-06-Working-with-LLMs-and-Model-APIs/07-What-is-an-inference-API.md)
- [BONUS: What are async and await?](16-BONUS-What-are-async-and-await-and-why-do-they-matter-for-LLM-apps.md)
