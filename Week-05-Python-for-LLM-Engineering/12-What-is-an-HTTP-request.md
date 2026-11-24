# What is an HTTP request?

> Week 05 – Python for LLM Engineering · Question 12

## Short Answer
An HTTP request is a message a client (your Python script, a browser) sends to a server to ask for or send data. It consists of a **method** (GET, POST, ...), a **URL**, **headers** (metadata such as authentication and content type) and an optional **body** (for example JSON). The server replies with an HTTP **response** containing a **status code** (200 OK, 429 Too Many Requests, ...), headers and a body. Every hosted LLM call, including what the Anthropic SDK does for you, is an HTTPS request underneath.

## Detailed Explanation

**Anatomy of an LLM API request:**
```
POST /v1/messages HTTP/1.1                    <- method + path
Host: api.anthropic.com                       <- headers ...
x-api-key: $ANTHROPIC_API_KEY
anthropic-version: 2023-06-01
content-type: application/json

{"model": "claude-haiku-5-5", "max_tokens": 50,          <- body (JSON)
 "messages": [{"role": "user", "content": "Hi"}]}
```

**And the response:**
```
HTTP/1.1 200 OK                               <- status line
content-type: application/json
request-id: req_...

{"content": [{"type": "text", "text": "Hello!"}], "usage": {...}}
```

**Status code families:**

| Range | Meaning | LLM examples |
|---|---|---|
| 2xx | Success | 200 OK |
| 3xx | Redirect | rarely seen with APIs |
| 4xx | Client error (your request is the problem) | 400 bad request, 401 bad key, 404 wrong URL/model, 429 rate limited |
| 5xx | Server error | 500 internal, 503 unavailable, 529 overloaded (Anthropic-specific) |

**Other key ideas:**
- **HTTPS** = HTTP over TLS encryption. Always use it for API keys.
- **Stateless:** each request stands alone. That's why you resend the whole chat history on each LLM call.
- **Timeouts:** always set one. LLM responses can take many seconds.
- **Streaming:** the server can send the body incrementally (Server-Sent Events), which LLM APIs use to stream tokens.

## Example
**Analogy:** an HTTP request is like posting a letter. The envelope has an address (URL), stamps and labels (headers), and the letter inside (body). The reply letter starts with a stamp saying "delivered", "wrong address" or "try later" (the status code).

```python
# pip install httpx
import os
import httpx

resp = httpx.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],   # never hardcode keys
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    json={
        "model": "claude-haiku-5-5",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say hello in French."}],
    },
    timeout=30.0,
)
print(resp.status_code)                       # 200
print(resp.headers["content-type"])           # application/json
print(resp.json()["content"][0]["text"])      # e.g. Bonjour !
```

In practice you'd use the official SDK (`anthropic.Anthropic().messages.create(...)`), which builds this same request and adds retries and typed errors. Seeing the raw request once helps you debug.

## Common Mistakes / Misconceptions
- **No timeout.** A request can hang forever. Always set one.
- **Ignoring the status code** and calling `.json()` on an error page. Check `resp.status_code` or call `resp.raise_for_status()`.
- **Putting secrets in the URL** (query strings end up in logs). Use headers.
- **Expecting the server to "remember" the conversation.** HTTP is stateless, so you send context every time.

## Key Takeaways
- A request = method + URL + headers + optional body. A response = status code + headers + body.
- 4xx means fix your request. 5xx (and 429) are usually worth retrying.
- LLM APIs are JSON-over-HTTPS. SDKs are convenient wrappers around them.
- Always use HTTPS, timeouts and header-based auth.

## Related Questions
- [What are GET and POST requests?](13-What-are-GET-and-POST-requests.md)
- [How do you consume a REST API using Python?](14-How-do-you-consume-a-REST-API-using-Python.md)
- [What is an inference API?](../Week-06-Working-with-LLMs-and-Model-APIs/07-What-is-an-inference-API.md)
- [What is an API key?](../Week-06-Working-with-LLMs-and-Model-APIs/08-What-is-an-API-key.md)
