# How does FastAPI help in AI applications?

> Week 11 – Production LLM Applications · Question 3

## In One Line
FastAPI is a Python tool for building web APIs quickly, and it is a great fit for AI apps because it checks inputs, handles slow model calls well and can stream answers.

## Simple Explanation
Your AI code needs a **web API** (an address like `/chat` that other apps can send requests to). **FastAPI** is a popular Python framework for building these.

Why it suits AI apps:

1. **It checks the input for you.** You describe what a request should look like (using a library called **Pydantic**). If someone sends an empty or huge question, FastAPI rejects it **before** you spend money on the model.
2. **It handles waiting well.** A model call can take several seconds. FastAPI supports **async** code, which means while one request is waiting for the model, the server can work on other requests.
3. **It can stream.** With `StreamingResponse`, the answer appears word by word, like in ChatGPT, so users don't stare at a blank screen.
4. **Free documentation.** It creates a test page at `/docs` automatically.
5. **It's Python.** It sits right next to the Anthropic SDK, NumPy and vector database tools.

```
Request → FastAPI checks input → your code calls the LLM → JSON or streamed answer
```

## Real-Life Example
FastAPI is like a good waiter. It checks your order is complete before sending it to the kitchen (input checks). It serves other tables while your food cooks (async). And it brings dishes out one by one instead of waiting for the whole meal (streaming).

## Code Example
A small chat endpoint. (This needs `fastapi`, `uvicorn`, `anthropic` and an `ANTHROPIC_API_KEY`, so the output below is an example.)

```python
import anthropic
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()
client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from the environment

class ChatIn(BaseModel):                # what a valid request must look like
    message: str = Field(min_length=1, max_length=4000)

@app.post("/chat")
def chat(body: ChatIn):                 # FastAPI has already checked `body`
    resp = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        system="You are a helpful tutor.",
        messages=[{"role": "user", "content": body.message}],
    )
    return {"reply": resp.content[0].text}

# Run:  uvicorn app:app
# POST /chat {"message": "What is a token?"}  -> e.g. {"reply": "A token is a small piece of text..."}
# POST /chat {"message": ""}                  -> 422 error, no model call, no cost
```

**What happened:** FastAPI checked the request first, so the empty message was rejected for free. Only valid requests reached the model.

## Watch Out
- Async helps with **waiting**, not with heavy computing. Running a big model on your own machine is still slow.
- If you write `async def`, use the async client (`anthropic.AsyncAnthropic()`). A normal client inside `async def` blocks the server. Using a plain `def`, as above, is fine.

## Remember
- FastAPI turns your Python AI code into a web API.
- Input checking is your **first and cheapest** safety check.
- Async handles many slow model calls; streaming makes answers feel fast.

## Related Questions
- [How do you design an LLM-powered application?](01-How-do-you-design-an-LLM-powered-application.md)
- [What is response streaming?](../Week-06-Working-with-LLMs-and-Model-APIs/14-What-is-response-streaming.md)
- [How do you consume a REST API using Python?](../Week-05-Python-for-LLM-Engineering/14-How-do-you-consume-a-REST-API-using-Python.md)
- [What is rate limiting?](09-What-is-rate-limiting.md)
