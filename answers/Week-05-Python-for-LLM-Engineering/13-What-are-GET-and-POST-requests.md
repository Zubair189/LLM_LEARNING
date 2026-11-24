# What are GET and POST requests?

> Week 05 – Python for LLM Engineering · Question 13

## In One Line
**GET** asks a server to **give you** data, and **POST** **sends** data to the server so it can do something with it.

## Simple Explanation
GET and POST are the two most common HTTP **methods**. A method is the "verb" of a request: it says what kind of action you want.

**GET = read.**
1. Used to fetch something, like a list of available models.
2. Any extra details go in the URL, like `?limit=5`.
3. It should not change anything on the server, so repeating it is safe.

**POST = send.**
1. Used to submit data or ask the server to do work.
2. The data goes in the **body** of the request, usually as JSON.
3. It can change things, so repeating it might do the work twice.

**Why LLM APIs use POST for chat:** prompts can be long, structured and private. They fit in a body, not in a URL. URLs also get saved in logs, which is bad for private data.

Other methods you may meet: `PUT` (replace), `PATCH` (update part of something) and `DELETE` (remove).

**Link to LLMs:** `POST /v1/messages` asks Claude for a reply. `GET /v1/models` lists the models you can use.

## Real-Life Example
GET is reading the menu on the wall: you just look and nothing changes. POST is handing your order to the waiter: you send details, and the kitchen starts cooking something new.

## Code Example
Talking to a local Ollama server.

```python
# pip install httpx   (needs Ollama running: ollama run llama3.2)
import httpx

BASE = "http://localhost:11434"                       # Ollama's local address

tags = httpx.get(f"{BASE}/api/tags", timeout=10).json()   # GET: just read the model list
print([m["name"] for m in tags["models"]])

resp = httpx.post(f"{BASE}/api/chat", timeout=60, json={  # POST: send a chat request
    "model": "llama3.2",                              # which model
    "messages": [{"role": "user", "content": "One word: capital of Japan?"}],
    "stream": False,                                  # reply all at once
})
print(resp.json()["message"]["content"])              # the model's answer

# Output (e.g.):
# ['llama3.2:latest']
# Tokyo
```

**What happened:** the GET only read information. The POST sent our question in a JSON body and the model did work to answer it.

## Watch Out
- POST is **not** automatically secure. Both GET and POST are only private over **HTTPS**.
- Never put API keys or private prompts in a GET URL. URLs are often logged.

## Remember
- **GET** = read, data in the URL.
- **POST** = send, data in the body.
- LLM chat calls are **POST** requests with JSON.

## Related Questions
- [What is an HTTP request?](12-What-is-an-HTTP-request.md)
- [How do you consume a REST API using Python?](14-How-do-you-consume-a-REST-API-using-Python.md)
- [What is Ollama?](../Week-06-Working-with-LLMs-and-Model-APIs/05-What-is-Ollama.md)
