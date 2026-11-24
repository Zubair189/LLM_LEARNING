# What is the difference between local and cloud LLMs?

> Week 06 – Working with LLMs and Model APIs · Question 6

## In One Line
A **local LLM** runs on your own computer, while a **cloud LLM** runs on a company's servers and you use it over the internet.

## Simple Explanation
**Local LLM**
- Runs on hardware you own, using tools like Ollama.
- Uses **open-weight** models (models whose files you can download), like Llama.
- Good: your data stays private, it works offline, and there is no fee per use.
- Not so good: limited by your computer's power, so models are usually smaller and less clever.

**Cloud LLM**
- Runs in a provider's data centre, like Anthropic's Claude.
- You send your prompt over the internet through an **API**.
- Good: the strongest models, very fast hardware, and no setup or maintenance.
- Not so good: you pay per **token** (small piece of text), your data goes to another company, and there are **rate limits** (a cap on how much you can send per minute).

**How to choose:**
1. Private or sensitive data, or no internet → **local**.
2. Hard reasoning, long documents, or a production app → **cloud**.
3. Both → **hybrid**: send private or easy jobs to a local model and hard jobs to the cloud.

**Link to LLMs:** many real apps use both and pick per request.

## Real-Life Example
Local is cooking at home: you own the kitchen and your ingredients stay private, but you're limited by your stove. Cloud is ordering from a restaurant: a better chef and no washing up, but you pay per dish and the food is made somewhere else.

## Code Example
A tiny "router" that keeps private data local.

```python
import requests, anthropic

def ask_local(prompt):                                    # runs on our own machine
    r = requests.post("http://localhost:11434/api/chat", timeout=120, json={
        "model": "llama3.2", "stream": False,
        "messages": [{"role": "user", "content": prompt}]})
    return r.json()["message"]["content"]

def ask_cloud(prompt):                                    # runs on Anthropic's servers
    resp = anthropic.Anthropic().messages.create(
        model="claude-sonnet-5-5", max_tokens=1024,
        messages=[{"role": "user", "content": prompt}])
    return resp.content[0].text

def ask(prompt, is_private):
    return ask_local(prompt) if is_private else ask_cloud(prompt)  # choose where to send it

print(ask("Summarise: patient John, 54, reports chest pain.", is_private=True))

# Output (e.g.):
# A 54-year-old patient named John reports chest pain.
```

**What happened:** because the text had private health details, it was sent to the local model and never left the computer.

## Watch Out
- "Local is free" is not quite true. Hardware, electricity and your time all cost money.
- Don't judge local models by comparing a tiny one with a top cloud model on a hard task. Match the model size to the job.

## Remember
- **Local** = private, offline, limited by your hardware.
- **Cloud** = strongest models, pay per token, data leaves your machine.
- **Hybrid** setups use both.

## Related Questions
- [What is Ollama?](05-What-is-Ollama.md)
- [What is an inference API?](07-What-is-an-inference-API.md)
- [How do you manage API costs and rate limits?](17-How-do-you-manage-API-costs-and-rate-limits.md)
- [What is a foundation model?](02-What-is-a-foundation-model.md)
