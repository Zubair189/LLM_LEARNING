# Week 05 – Python for LLM Engineering

## Overview
This week covers the everyday Python you need to build LLM apps. You start with the basics: data types (lists, tuples, dictionaries), functions, classes, modules and errors. Next come the data tools: JSON, NumPy (fast number arrays, used for embeddings) and Pandas (tables of data). Then you learn how Python talks to other computers: HTTP requests, GET vs POST, using REST APIs, and handling API errors. Two bonus topics cover `async`/`await` (running many LLM calls at once) and virtual environments (keeping each project's packages separate). By the end you should be able to write a small, tidy Python program that calls an LLM, reads its JSON answer safely, and keeps working when the network is slow or busy.

## Questions
1. [Why is Python widely used in AI development?](01-Why-is-Python-widely-used-in-AI-development.md)
2. [What are Python data types?](02-What-are-Python-data-types.md)
3. [What is the difference between a list and a tuple?](03-What-is-the-difference-between-a-list-and-a-tuple.md)
4. [What is a dictionary?](04-What-is-a-dictionary.md)
5. [What are functions?](05-What-are-functions.md)
6. [What is object-oriented programming?](06-What-is-object-oriented-programming.md)
7. [What are modules and packages?](07-What-are-modules-and-packages.md)
8. [How do Python exceptions work?](08-How-do-Python-exceptions-work.md)
9. [What is JSON?](09-What-is-JSON.md)
10. [What is NumPy?](10-What-is-NumPy.md)
11. [What is Pandas?](11-What-is-Pandas.md)
12. [What is an HTTP request?](12-What-is-an-HTTP-request.md)
13. [What are GET and POST requests?](13-What-are-GET-and-POST-requests.md)
14. [How do you consume a REST API using Python?](14-How-do-you-consume-a-REST-API-using-Python.md)
15. [How do you handle API errors?](15-How-do-you-handle-API-errors.md)
16. **BONUS** – [What are async and await, and why do they matter for LLM apps?](16-BONUS-What-are-async-and-await-and-why-do-they-matter-for-LLM-apps.md)
17. **BONUS** – [What are virtual environments and how do you manage dependencies?](17-BONUS-What-are-virtual-environments-and-how-do-you-manage-dependencies.md)

> **Overlap notes:** Q12 explains the *parts* of any request and response. Q13 compares two *methods* (GET and POST). Q14 is the full *step-by-step workflow*. Q8 explains Python errors in general, and Q15 uses them to handle API failures and retries.

## Practical Exercise (1–2 hours): "Review Sentiment Batch Analyser"
Build `analyse.py` in a fresh virtual environment (`uv init` or `python -m venv`):
1. **Data:** make `reviews.csv` with about 20 short product reviews (columns `id`, `text`) and load it with Pandas.
2. **LLM call:** write `classify(text) -> dict` that asks a model (`claude-sonnet-5-5` via the Anthropic SDK, or `llama3.2` via Ollama's `POST /api/chat`) to reply with JSON like `{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}`.
3. **Safety:** read the reply with `json.loads` inside `try/except`. If the reply is bad, raise your own `LLMParseError` and try once more. If the API is busy (`RateLimitError` / `APIConnectionError`), wait longer each time and retry.
4. **Structure:** put the model call inside a small class (`ClaudeProvider` / `OllamaProvider`), each with the same `.classify()` method.
5. **Speed (stretch):** switch to `AsyncAnthropic` + `asyncio.gather` with a `Semaphore(5)` and compare the total time with the normal version.
6. **Report:** add the results as new columns, then print the sentiment counts (`value_counts`), the average confidence, and the total input/output tokens from `resp.usage`. Save `results.jsonl`.

**Done when:** the script runs from start to finish, never crashes on a bad reply, and prints a summary table.

## Explain Aloud
Practise explaining each of these in under 60 seconds, without notes:
1. **Changeable vs fixed values**: why a tuple can be a dictionary key but a list can't, and why `def f(history=[])` is a bug.
2. **The journey of an LLM API call**: Python dict → JSON text → HTTPS POST with headers → status code → JSON reply → checked Python data.
3. **When to retry**: which errors you retry (timeouts, 429, 5xx) and which you don't (400, 401, 404), and why waiting a bit longer each time helps.
