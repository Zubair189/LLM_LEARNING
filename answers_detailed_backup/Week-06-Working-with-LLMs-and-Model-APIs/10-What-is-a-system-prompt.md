# What is a system prompt?

> Week 06 – Working with LLMs and Model APIs · Question 10

> Note: The **system prompt** is set by the developer and defines persistent behaviour for the whole conversation; the **user prompt** ([Q11](11-What-is-a-user-prompt.md)) is the per-turn request, usually from the end user.

## Short Answer
A system prompt is a developer-supplied instruction that sets the model's role, rules, tone, knowledge context and output format for an entire conversation, separate from the user's messages. In the Anthropic API it is the top-level `system` parameter; in OpenAI-style and Ollama APIs it is a message with `"role": "system"`. Models are trained to give it high priority, so it is the main place to define how your application behaves.

## Detailed Explanation
```
 ┌────────────── system (developer) ───────────────┐
 │ "You are a support bot for Acme. Be concise.    │
 │  Only answer about Acme products. Reply in the  │
 │  user's language. Today's date: 2026-11-24."    │
 └─────────────────────────────────────────────────┘
 user:      "How do I reset my router?"
 assistant: "..."
 user:      "And the Wi-Fi password?"
 assistant: "..."
```

Typical contents:
- **Role/persona** – "You are a senior Python reviewer."
- **Rules & boundaries** – scope, safety constraints, what to refuse.
- **Style** – tone, length, language.
- **Output format** – Markdown, JSON shape, citations.
- **Context** – company facts, current date, retrieved documents, tool usage guidance.

| | System prompt | User prompt |
|---|---|---|
| Author | Developer | End user (or your code) |
| Scope | Whole conversation | One turn |
| Changes | Rarely (versioned with code) | Every request |
| Trust level | Trusted | Untrusted input |

Important limits: a system prompt is strong guidance, not a security boundary. Determined users can attempt prompt injection, so enforce critical rules in code too (validation, permission checks). Long, stable system prompts can also be cached with prompt caching to reduce cost and latency.

## Example
**Analogy:** The system prompt is the job description and company handbook given to a new employee on day one; user prompts are the individual customer requests they handle afterwards.

```python
import anthropic

client = anthropic.Anthropic()
SYSTEM = ("You are a pirate-themed coding tutor. Always answer in under 40 words "
          "and end with 'Arrr!'.")

resp = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=200,
    system=SYSTEM,                                   # top-level, not inside messages
    messages=[{"role": "user", "content": "What is a Python list?"}],
)
print(resp.content[0].text)
# Expected (wording varies): "A list be an ordered, changeable collection, like [1, 2, 3]. Arrr!"
```

Equivalent with Ollama (system as a message):
```python
import requests
r = requests.post("http://localhost:11434/api/chat", json={
    "model": "llama3.2", "stream": False,
    "messages": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "What is a Python list?"}]}, timeout=120)
print(r.json()["message"]["content"])
```

## Common Mistakes / Misconceptions
- **Putting the system prompt inside `messages` for the Anthropic API.** Use the `system` parameter; `messages` only accepts `user` and `assistant` roles.
- **Treating it as secret.** Users can often coax a model into revealing it; don't put keys or confidential data there.
- **Relying on it as the only safety control.** Validate outputs and enforce permissions in code.
- **Stuffing per-request data into it** that belongs in the user turn, making caching and debugging harder.

## Key Takeaways
- System prompt = developer instructions that shape the whole conversation.
- Use it for role, rules, style, format and stable context.
- Anthropic: `system=` parameter; OpenAI/Ollama: `role: "system"` message.
- Strong guidance, not a hard security guarantee.

## Related Questions
- [What is a user prompt?](11-What-is-a-user-prompt.md)
- [What is prompt engineering?](09-What-is-prompt-engineering.md)
- [What is few-shot prompting?](18-BONUS-What-is-few-shot-prompting.md)
- [What is a dictionary?](../Week-05-Python-for-LLM-Engineering/04-What-is-a-dictionary.md)
