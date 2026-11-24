# What is the role of an API gateway?

> Week 11 – Production LLM Applications · Question 2

## In One Line
An API gateway is the **single front door** to your app: every request passes through it to be checked and sent to the right place.

## Simple Explanation
An **API** is a way for one program to talk to another over the internet. Your app may have several small services behind it, for example a chat service and a search service.

Without a gateway, each service has to check users, count requests and write logs by itself, and they will all do it a bit differently. A **gateway** does these common jobs in one place:

1. **Who are you?** It checks the user's login or key (**authentication**).
2. **Too many requests?** It applies **rate limits**, a cap on how often someone can call.
3. **Too big?** It rejects huge requests before they become huge, expensive prompts.
4. **Where should this go?** It **routes** the request, e.g. `/chat` to the chat service.
5. **Write it down.** It logs each request with an ID so it can be traced later.

**Why it matters for LLM apps:** every model call costs money. The gateway makes sure only allowed users can spend your tokens. It also keeps your secret LLM provider key on the server, never in the browser.

```
Browser / App → API Gateway (login, limits, size, logs) → chat service → LLM
```

Common gateways include Kong, NGINX and AWS API Gateway.

## Real-Life Example
A hotel reception desk. Guests don't wander into the kitchen or the manager's office. Reception checks who you are, sends you to the right place, and writes down who came in.

## Code Example
The main checks a gateway does, written as one plain Python function.

```python
VALID_KEYS = {"team-a-key", "team-b-key"}   # keys we gave to our client apps
MAX_SIZE = 2000                             # biggest request we accept (characters)

def gateway(api_key, body, path):
    if api_key not in VALID_KEYS:           # 1. who are you?
        return 401, "unauthorized"
    if len(body) > MAX_SIZE:                # 2. is the request too big?
        return 413, "request too large"
    if path == "/chat":                     # 3. send it to the right service
        return 200, "sent to chat service"
    if path == "/search":
        return 200, "sent to search service"
    return 404, "unknown path"

print(gateway("team-a-key", "Hello", "/chat"))     # (200, 'sent to chat service')
print(gateway("wrong-key", "Hello", "/chat"))      # (401, 'unauthorized')
print(gateway("team-b-key", "x" * 5000, "/search"))  # (413, 'request too large')
```

**What happened:** only the request with a valid key and a normal size got through. The numbers (401, 413) are standard web error codes.

## Watch Out
- A gateway does **not** read the meaning of the prompt, so it can't stop prompt injection. You still need guardrails for that.
- Never call the LLM provider straight from the browser. That exposes your API key to anyone.

## Remember
- The gateway is the **one controlled entry point**.
- It handles login, rate limits, size limits, routing and logs.
- For LLM apps it protects your **keys and your budget**.

## Related Questions
- [What is rate limiting?](09-What-is-rate-limiting.md)
- [How do you protect API keys?](08-How-do-you-protect-API-keys.md)
- [How do you design an LLM-powered application?](01-How-do-you-design-an-LLM-powered-application.md)
- [What is an API key?](../Week-06-Working-with-LLMs-and-Model-APIs/08-What-is-an-API-key.md)
