# What is Ollama?

> Week 06 – Working with LLMs and Model APIs · Question 5

## In One Line
Ollama is a free tool that lets you **download and run open LLMs on your own computer** with one command.

## Simple Explanation
Normally, running an LLM yourself is tricky. Ollama makes it easy.

How you use it:

1. **Install** Ollama from its website.
2. **Download a model:** `ollama pull llama3.2`.
3. **Chat in the terminal:** `ollama run llama3.2`.
4. **Use it from code:** Ollama runs a small local web server at `http://localhost:11434`. Your Python app sends requests there, just like it would to a cloud API.

Handy commands:
- `ollama list`: show downloaded models.
- `ollama ps`: show models currently loaded in memory.
- `ollama rm <model>`: delete a model.

**How do big models fit on a laptop?** Ollama uses **quantized** models. Quantizing means storing the model's numbers with less precision, so the model takes much less memory, with only a small loss in quality.

Main API addresses:
- `POST /api/chat` for chat messages.
- `POST /api/generate` for a single prompt.
- `POST /api/embed` for embeddings.

**Link to LLMs:** Ollama is great for privacy (your data never leaves your machine), working offline, and free experiments.

## Real-Life Example
Ollama is like a video game launcher for AI models. You pick a model, it downloads it, and you press "play". Everything runs on your own machine.

## Code Example
```python
# pip install requests   (Ollama must be running and llama3.2 pulled)
import requests

resp = requests.post("http://localhost:11434/api/chat", json={   # local server
    "model": "llama3.2",                                     # which local model
    "messages": [
        {"role": "system", "content": "Answer in one short sentence."},
        {"role": "user", "content": "What is a token?"},
    ],
    "stream": False,                                         # one complete reply
}, timeout=120)
resp.raise_for_status()                                      # stop if there was an error
print(resp.json()["message"]["content"])                     # the model's answer

# Output (e.g.):
# A token is a small piece of text, like a word or part of a word.
```

**What happened:** our Python code talked to a model running on our own computer, using the same kind of JSON request a cloud API uses.

## Watch Out
- Without `"stream": False`, Ollama sends the answer in many small pieces and `resp.json()` fails.
- Small local models (1–3 billion parameters) are weaker than big cloud models at hard reasoning.

## Remember
- Ollama = **run open LLMs locally**, easily.
- Local API at **`localhost:11434`**.
- Good for **privacy, offline use and free testing**.

## Related Questions
- [What is the difference between local and cloud LLMs?](06-What-is-the-difference-between-local-and-cloud-LLMs.md)
- [What is a model repository?](04-What-is-a-model-repository.md)
- [What is JSON mode?](16-What-is-JSON-mode.md)
- [How do you consume a REST API using Python?](../Week-05-Python-for-LLM-Engineering/14-How-do-you-consume-a-REST-API-using-Python.md)
