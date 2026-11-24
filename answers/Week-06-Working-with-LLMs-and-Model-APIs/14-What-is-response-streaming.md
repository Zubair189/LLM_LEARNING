# What is response streaming?

> Week 06 – Working with LLMs and Model APIs · Question 14

## In One Line
Streaming means the model's answer is **sent to you bit by bit as it is written**, instead of all at once at the end.

## Simple Explanation
An LLM writes its answer one token (small piece of text) at a time. A long answer can take several seconds.

**Without streaming**, you wait until the whole answer is finished, then get it in one go. The screen stays blank meanwhile.

**With streaming**, the server sends each small piece as soon as it is ready. Your app shows the text appearing word by word, like someone typing.

Important points:

1. **It feels much faster.** The first words arrive in a fraction of a second.
2. **The total time is about the same.** Streaming doesn't make the model itself faster.
3. **You still collect the full text** if you want to save it or read it as JSON.

How it is sent:
- **Cloud APIs** (like Anthropic) use **Server-Sent Events (SSE)**, a way for a server to keep sending small messages over one open connection.
- **Ollama** sends one small JSON object per line.

**Link to LLMs:** this is why chatbots show text "typing out". In the Anthropic SDK you use `client.messages.stream(...)`.

## Real-Life Example
Without streaming is like getting a letter only after the whole thing is written. Streaming is like watching someone write on a whiteboard: you can start reading straight away.

## Code Example
```python
import anthropic

client = anthropic.Anthropic()                                # reads ANTHROPIC_API_KEY

with client.messages.stream(                                  # open a streaming request
    model="claude-sonnet-5-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a 3-line poem about Python."}],
) as stream:
    for text in stream.text_stream:                           # each small piece as it arrives
        print(text, end="", flush=True)                       # print without a new line, right away
    final = stream.get_final_message()                        # the complete message at the end

print("\noutput tokens:", final.usage.output_tokens)          # tokens used (for cost)

# Output (e.g., the poem appears gradually):
# Indented lines, so clean and neat,
# ...
# output tokens: 28
```

**What happened:** the poem appeared on screen piece by piece. At the end we still got the full message and its token count.

## Watch Out
- Don't try to read JSON from half-finished pieces. Collect the full text first.
- Use `flush=True` when printing, or the text may appear all at once anyway.

## Remember
- Streaming = **answer arrives piece by piece**.
- It **feels** faster; total time is similar.
- Great for **chat apps** and long answers.

## Related Questions
- [What is an inference API?](07-What-is-an-inference-API.md)
- [What are async and await, and why do they matter for LLM apps?](../Week-05-Python-for-LLM-Engineering/16-BONUS-What-are-async-and-await-and-why-do-they-matter-for-LLM-apps.md)
- [What is an HTTP request?](../Week-05-Python-for-LLM-Engineering/12-What-is-an-HTTP-request.md)
- [What is Ollama?](05-What-is-Ollama.md)
