# How do you consume a REST API using Python?

> Week 05 – Python for LLM Engineering · Question 14

## Short Answer
To consume a REST API, you send HTTP requests to its resource URLs with an HTTP client library, usually `requests` or `httpx` (which also supports async). You include authentication headers and a JSON body where needed, check the status code, and parse the JSON response into Python dicts. For production code, wrap this in a small client function with timeouts, error handling and retries. When an official SDK exists, as it does for Anthropic, OpenAI and Hugging Face, prefer it: it does all of this for you and returns typed objects.

## Detailed Explanation

**The standard recipe:**
1. **Read the docs:** base URL, endpoint, method, required headers, request and response schema, rate limits.
2. **Load secrets from the environment** (`os.environ["..."]` or a `.env` file), never from source code.
3. **Build the request:** method + URL + headers + `params` (query) or `json` (body).
4. **Send it with a timeout.**
5. **Check the status:** `resp.raise_for_status()`, or branch on `resp.status_code`.
6. **Parse:** `resp.json()`, then validate the shape (e.g. with Pydantic).
7. **Handle failures:** retry 429/5xx with backoff, and surface 4xx errors clearly.

**Choosing a client:**

| Option | When to use |
|---|---|
| `requests` | Simple synchronous scripts; very widely known |
| `httpx` | Modern; sync **and** async; HTTP/2; used inside many LLM SDKs |
| Official SDK (`anthropic`, `openai`, `huggingface_hub`) | Best choice when available: typed responses, retries, streaming helpers |

**Reuse connections:** create one `httpx.Client()` or `requests.Session()` and reuse it. This is faster (keep-alive) and lets you set shared headers once.

## Example
**Analogy:** consuming a REST API is like ordering from a restaurant's online menu. Read the menu (docs), place the order in the expected format (request), show your membership card (API key), check whether the order was accepted (status code), then unpack the delivery (parse the JSON).

**Option A: raw REST with `httpx` (against a local Ollama server):**
```python
import httpx

def ollama_chat(prompt: str, model: str = "llama3.2") -> str:
    with httpx.Client(base_url="http://localhost:11434", timeout=60) as client:
        resp = client.post("/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        })
        resp.raise_for_status()                 # raises on 4xx/5xx
        return resp.json()["message"]["content"]

print(ollama_chat("Give one synonym for 'fast'."))   # e.g. Quick
```

**Option B: the official SDK (recommended for hosted APIs):**
```python
import anthropic

client = anthropic.Anthropic(max_retries=3, timeout=60.0)  # key from ANTHROPIC_API_KEY
resp = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=200,
    system="You are a helpful assistant. Answer in one sentence.",
    messages=[{"role": "user", "content": "What is a REST API?"}],
)
print(resp.content[0].text)
# e.g. A REST API exposes resources at URLs that clients manipulate with HTTP methods like GET and POST.
print(resp.usage.input_tokens, resp.usage.output_tokens)   # e.g. 28 24
```

The SDK sends the same kind of POST request as Option A, but it adds authentication headers, automatic retries for 429/5xx errors, typed exceptions and helpers like `.usage`.

## Common Mistakes / Misconceptions
- **Hardcoding API keys** in code or notebooks that get committed. Use environment variables and secret managers.
- **No timeout and no status check**, which leads to hangs or confusing `KeyError`s when the body is actually an error message.
- **Creating a new client per request in a hot loop.** Reuse one client or session.
- **Trusting the response shape.** APIs evolve. Validate fields and pin SDK/API versions.

## Key Takeaways
- Pattern: auth from env → request with timeout → check status → parse → validate.
- Use `httpx` or `requests` for generic REST, and the official SDK when one exists.
- Reuse clients, and use async (`httpx.AsyncClient`, `anthropic.AsyncAnthropic`) for many concurrent calls.
- Retries with backoff belong in every production API client.

## Related Questions
- [What is an HTTP request?](12-What-is-an-HTTP-request.md)
- [How do you handle API errors?](15-How-do-you-handle-API-errors.md)
- [What is an inference API?](../Week-06-Working-with-LLMs-and-Model-APIs/07-What-is-an-inference-API.md)
- [BONUS: What are async and await?](16-BONUS-What-are-async-and-await-and-why-do-they-matter-for-LLM-apps.md)
