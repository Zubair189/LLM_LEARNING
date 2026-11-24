# Week 05 – Python for LLM Engineering

## Overview
This week covers the Python you need every day as an LLM engineer. You'll start with core data types (lists, tuples, dicts), functions, classes, modules and exceptions. Then you'll move to the data tools of the AI stack (JSON, NumPy for embeddings, Pandas for datasets and evaluation logs) and finish with talking to the outside world: HTTP requests, GET vs POST, consuming REST APIs and handling API errors robustly. Two bonus topics cover `async`/`await` for concurrent LLM calls and virtual environments for reproducible projects. By the end of the week you should be able to write a small, well-structured Python program that calls a hosted or local LLM, parses its JSON output safely, and survives timeouts and rate limits.

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

> **Overlap notes:** Q12 (HTTP request) covers the *anatomy* of any request and response. Q13 compares two *methods*. Q14 is the end-to-end *workflow*. Q8 (exceptions) is the general Python mechanism, and Q15 applies it to API failures with a retry policy.

## Practical Exercise (1–2 hours): "Review Sentiment Batch Analyser"
Build `analyse.py` in a fresh virtual environment (`uv init` or `python -m venv`):
1. **Data:** create `reviews.csv` with about 20 short product reviews (columns `id`, `text`) and load it with Pandas.
2. **LLM call:** write `classify(text) -> dict` that asks a model (either `claude-haiku-5-5` via the Anthropic SDK, or `llama3.2` via Ollama's `POST /api/chat`) to return JSON like `{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}`.
3. **Robustness:** parse with `json.loads` inside `try/except`. Raise a custom `LLMParseError` on bad output and retry once. Catch `RateLimitError`/`APIConnectionError` with exponential backoff.
4. **Structure:** put the provider behind a small class (`ClaudeProvider` / `OllamaProvider`) with the same `.classify()` method.
5. **Concurrency (stretch):** switch to `AsyncAnthropic` + `asyncio.gather` with a `Semaphore(5)` and compare wall-clock time with the sync version.
6. **Report:** add the results as columns, then print the sentiment counts (`value_counts`), the mean confidence, and total input/output tokens from `resp.usage`. Save `results.jsonl`.

**Done when:** the script runs end-to-end, never crashes on a malformed reply, and prints a summary table.

## Explain Aloud
Practise explaining each of these in under 60 seconds, without notes:
1. **Mutable vs immutable**: why a tuple can be a dict key and a list can't, and why `def f(history=[])` is a bug.
2. **The life of an LLM API call**: Python dict → JSON body → HTTPS POST with headers → status code → JSON response → validated Python object.
3. **Retry strategy**: which errors you retry (timeouts, 429, 5xx) and which you don't (400, 401, 404), and why exponential backoff with jitter works.
