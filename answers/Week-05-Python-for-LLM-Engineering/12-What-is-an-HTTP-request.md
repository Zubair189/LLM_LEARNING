# What is an HTTP request?

> Week 05 – Python for LLM Engineering · Question 12

## In One Line
An HTTP request is a **message your program sends to a server** over the internet, and the server sends back a response.

## Simple Explanation
HTTP is the language programs use to talk on the web. Your code is the **client** (the one asking) and the other computer is the **server** (the one answering).

A request has four parts:

1. **Method:** what you want to do, for example `GET` (fetch) or `POST` (send).
2. **URL:** the address, like `https://api.anthropic.com/v1/messages`.
3. **Headers:** extra labels, such as your API key and "this body is JSON".
4. **Body** (optional): the data you send, usually JSON.

The server replies with a **response** that has a **status code**, headers and a body. Status codes tell you what happened:

- **2xx** (like 200): success.
- **4xx**: your request has a problem. 401 = bad key, 429 = too many requests.
- **5xx**: the server has a problem; trying again later often works.

**HTTPS** is HTTP with encryption, so nobody can read your key on the way.

**Link to LLMs:** every call to a cloud LLM is an HTTPS request. The server doesn't remember past calls, so you send the whole chat history every time.

## Real-Life Example
Sending a request is like posting a letter. The envelope has an address (URL) and labels (headers), and the letter inside is the body. The reply comes back stamped "delivered", "wrong address" or "try later" (the status code).

## Code Example
```python
# pip install httpx
import os
import httpx

resp = httpx.post(                                        # method: POST
    "https://api.anthropic.com/v1/messages",              # URL
    headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],     # key from the environment
        "anthropic-version": "2023-06-01",                # API version header
        "content-type": "application/json",               # body is JSON
    },
    json={"model": "claude-sonnet-5-5", "max_tokens": 1024,   # body
          "messages": [{"role": "user", "content": "Say hello in French."}]},
    timeout=30.0,                                         # don't wait forever
)
print(resp.status_code)                                   # status code
print(resp.json()["content"][0]["text"])                  # read the reply

# Output (e.g.):
# 200
# Bonjour !
```

**What happened:** we built the request by hand and got a 200 (success) plus the answer. The official `anthropic` SDK sends this same request for you.

## Watch Out
- Always set a **timeout**. Without one, a request can hang forever.
- Check the status code before reading the body. An error reply won't have the data you expect.

## Remember
- Request = **method + URL + headers + body**.
- Response = **status code + headers + body**.
- 2xx = OK, 4xx = your problem, 5xx = server problem.

## Related Questions
- [What are GET and POST requests?](13-What-are-GET-and-POST-requests.md)
- [How do you consume a REST API using Python?](14-How-do-you-consume-a-REST-API-using-Python.md)
- [What is an inference API?](../Week-06-Working-with-LLMs-and-Model-APIs/07-What-is-an-inference-API.md)
- [What is an API key?](../Week-06-Working-with-LLMs-and-Model-APIs/08-What-is-an-API-key.md)
