# What is response streaming?

> Week 06 – Working with LLMs and Model APIs · Question 14

## Short Answer
Response streaming means the API sends the model's output incrementally, token chunk by token chunk, as it is generated, instead of waiting until the whole answer is finished. It dramatically lowers *perceived* latency (time to first token) and is why chat UIs show text "typing" out. Hosted APIs typically stream using Server-Sent Events (SSE); Ollama streams newline-delimited JSON.

## Detailed Explanation
```
 Non-streaming:  request ──────── (wait 8 s) ────────► full answer
 Streaming:      request ─► "The" ─► " capital" ─► " of" ─► ... ─► done (8 s total)
                            ▲ first token after ~0.5 s
```

Total generation time is about the same; the user just starts reading much sooner.

How it works with the Anthropic API (`stream=True` under the hood):
- Server sends SSE events: `message_start` → `content_block_start` → many `content_block_delta` (text chunks) → `content_block_stop` → `message_delta` (stop_reason, final usage) → `message_stop`.
- The Python SDK's `client.messages.stream(...)` helper parses these and gives you `text_stream` and `get_final_message()`.

| | Non-streaming | Streaming |
|---|---|---|
| Time to first token | Full generation time | Fraction of a second |
| Code complexity | Simple | Slightly more (iterate chunks) |
| Easy to parse JSON? | Yes | Need to accumulate first |
| Long outputs | Risk of HTTP timeouts | Connection kept alive |
| Best for | Batch jobs, extraction | Chat UIs, long answers |

Things to handle: accumulate the full text if you need to save/parse it, handle errors that occur mid-stream, and still read final token usage for cost tracking.

## Example
**Analogy:** Non-streaming is receiving a letter only after it's fully written; streaming is watching someone write on a whiteboard in real time.

```python
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-sonnet-5-5",
    max_tokens=300,
    messages=[{"role": "user", "content": "Write a 3-line poem about Python."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)   # appears progressively
    final = stream.get_final_message()

print("\n---")
print("output tokens:", final.usage.output_tokens)
# Expected: the poem printed gradually, then e.g.
# ---
# output tokens: 28
```

Streaming from Ollama (newline-delimited JSON):
```python
import json, requests

with requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "Count to five."}],
        "stream": True}, stream=True, timeout=120) as r:
    for line in r.iter_lines():
        if line:
            chunk = json.loads(line)
            print(chunk["message"]["content"], end="", flush=True)
            if chunk.get("done"):
                break
# Expected: "1, 2, 3, 4, 5." printed progressively
```

## Common Mistakes / Misconceptions
- **"Streaming makes the model faster."** It improves perceived latency; total time is similar.
- **Parsing JSON from partial chunks.** Accumulate the full text (or use the final message) before `json.loads`.
- **Forgetting `flush=True`** – output is buffered and appears all at once.
- **Ignoring mid-stream errors** – connections can drop; handle exceptions and decide whether to retry.

## Key Takeaways
- Streaming sends output incrementally as it is generated.
- Big UX win via low time-to-first-token.
- Anthropic: `client.messages.stream(...)` + `text_stream`; Ollama: `"stream": true` NDJSON.
- Still collect the final message for full text and token usage.

## Related Questions
- [What is an inference API?](07-What-is-an-inference-API.md)
- [What are async and await, and why do they matter for LLM apps?](../Week-05-Python-for-LLM-Engineering/16-BONUS-What-are-async-and-await-and-why-do-they-matter-for-LLM-apps.md)
- [What is an HTTP request?](../Week-05-Python-for-LLM-Engineering/12-What-is-an-HTTP-request.md)
- [What is Ollama?](05-What-is-Ollama.md)
