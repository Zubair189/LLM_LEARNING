# What is Ollama?

> Week 06 – Working with LLMs and Model APIs · Question 5

## Short Answer
Ollama is a free, open-source tool for downloading and running open-weight LLMs (Llama, Qwen, Mistral, Gemma, etc.) on your own machine with one command, e.g. `ollama run llama3.2`. It packages quantized models, manages them like Docker images, and exposes a local REST API on `http://localhost:11434`, so your Python app can talk to a local model almost exactly like a cloud API.

## Detailed Explanation
Under the hood Ollama uses an inference engine based on llama.cpp and quantized model files (GGUF) so that multi-billion-parameter models fit in laptop RAM/VRAM.

```
 ollama pull llama3.2     →  downloads model to ~/.ollama/models
 ollama run llama3.2      →  interactive chat in the terminal
 ollama serve (auto)      →  REST API at http://localhost:11434
        │
        ├── POST /api/chat       (chat messages)
        ├── POST /api/generate   (raw prompt completion)
        ├── POST /api/embed      (embeddings)
        └── /v1/chat/completions (OpenAI-compatible endpoint)
```

Useful commands:

| Command | Purpose |
|---|---|
| `ollama pull qwen2.5:7b` | Download a model (tag = size/variant) |
| `ollama list` | Show installed models |
| `ollama ps` | Show models loaded in memory |
| `ollama rm <model>` | Delete a model |
| `Modelfile` + `ollama create` | Custom model with your own system prompt/parameters |

**Hardware rule of thumb (illustrative):** a 4-bit quantized model needs roughly 0.5–0.7 GB of memory per billion parameters, so a 3B model runs on most laptops while 70B needs a high-end GPU or lots of unified memory.

Ollama also supports structured output (`format="json"` or a JSON schema), tool calling and vision models, and there is an official `ollama` Python package.

## Example
**Analogy:** Ollama is like Docker for LLMs — `pull` an image, `run` it, and talk to it over a local port.

```bash
ollama run llama3.2 "Give me one tip for writing prompts."
```

```python
# Call the local REST API with requests (Ollama must be running, model pulled)
import requests

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": "Answer in one short sentence."},
            {"role": "user", "content": "What is a token?"},
        ],
        "stream": False,          # return one JSON object instead of a stream
        "options": {"temperature": 0.2},
    },
    timeout=120,
)
resp.raise_for_status()
print(resp.json()["message"]["content"])
# Expected (wording varies): "A token is a small chunk of text, like a word or part of a word, that the model processes."
```

## Common Mistakes / Misconceptions
- **Forgetting `"stream": False`.** By default `/api/chat` streams newline-delimited JSON; `resp.json()` then fails.
- **Expecting cloud-model quality from a 1–3B model.** Small local models are great for privacy and prototyping but weaker at complex reasoning.
- **Model not pulled / server not running.** A 404 "model not found" or connection refused usually means you need `ollama pull` or to start Ollama.
- **Exposing port 11434 to the internet.** The local API has no authentication by default.

## Key Takeaways
- Ollama runs open-weight LLMs locally with simple CLI commands.
- It serves a REST API at `localhost:11434` (`/api/chat`, `/api/generate`, `/api/embed`) plus an OpenAI-compatible endpoint.
- Uses quantized GGUF models so they fit consumer hardware.
- Ideal for offline work, privacy-sensitive data and cost-free experimentation.

## Related Questions
- [What is the difference between local and cloud LLMs?](06-What-is-the-difference-between-local-and-cloud-LLMs.md)
- [What is a model repository?](04-What-is-a-model-repository.md)
- [What is JSON mode?](16-What-is-JSON-mode.md)
- [How do you consume a REST API using Python?](../Week-05-Python-for-LLM-Engineering/14-How-do-you-consume-a-REST-API-using-Python.md)
