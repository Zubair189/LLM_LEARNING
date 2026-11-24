# What is a user prompt?

> Week 06 – Working with LLMs and Model APIs · Question 11

> Note: Contrast with the [system prompt](10-What-is-a-system-prompt.md): the system prompt is the developer's standing instructions; the user prompt is the specific request for this turn.

## Short Answer
A user prompt is the message in the `"user"` role that carries the actual request for a given turn — a question, a task, or data to process. It usually comes from the end user, but in pipelines your code often builds it from a template plus inputs like retrieved documents. The model answers it within the rules set by the system prompt, and each user message is paired with an `"assistant"` reply in the conversation history.

## Detailed Explanation
Chat APIs represent a conversation as an ordered list of messages:

```python
messages = [
    {"role": "user",      "content": "What's the capital of France?"},   # turn 1
    {"role": "assistant", "content": "Paris."},                          # model reply
    {"role": "user",      "content": "And its population?"},             # turn 2 (needs history)
]
```

Key points:
- **Alternating roles** – user and assistant messages alternate; the last message is normally `user`.
- **Statelessness** – the API does not remember turn 1; you resend the history so "its" can be resolved.
- **Content can be rich** – text, images, documents, or tool results (as content blocks).
- **Templating** – apps often wrap raw user input inside a template:

```
 template:  "Answer using only the context.\n<context>{docs}</context>\n<question>{q}</question>"
 user input: q = "What is the refund window?"
```

| | User prompt | System prompt |
|---|---|---|
| Role | `user` | `system` param / role |
| Frequency | Every turn | Once per conversation |
| Trust | Untrusted (may contain injection) | Trusted |
| Typical content | Question, data, task | Persona, rules, format |

Because user text is untrusted, wrap it in delimiters and never let it override your system rules or trigger privileged actions without checks.

## Example
**Analogy:** If the system prompt is the restaurant's menu and kitchen rules, the user prompt is each customer's individual order.

A minimal multi-turn chat loop that maintains history:

```python
import anthropic

client = anthropic.Anthropic()
history = []

def chat(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    resp = client.messages.create(
        model="claude-haiku-5-5", max_tokens=200,
        system="You are a helpful geography assistant. Be brief.",
        messages=history,
    )
    reply = resp.content[0].text
    history.append({"role": "assistant", "content": reply})
    return reply

print(chat("What's the capital of France?"))   # Expected: Paris.
print(chat("And its population?"))             # Expected: ~2.1 million (city proper), ~12 million metro
print(len(history))                            # Expected: 4
```

Building a templated user prompt safely:
```python
def build_prompt(question: str, docs: list[str]) -> str:
    context = "\n".join(f"<doc>{d}</doc>" for d in docs)
    return f"Use only the documents.\n{context}\n<question>{question}</question>"

print(build_prompt("Refund window?", ["Refunds within 30 days."]))
# Expected:
# Use only the documents.
# <doc>Refunds within 30 days.</doc>
# <question>Refund window?</question>
```

## Common Mistakes / Misconceptions
- **Sending only the latest user message** and wondering why the model "forgets" context.
- **Trusting user input** – "ignore previous instructions" is a classic injection attempt.
- **Two `user` messages in a row** by accident when appending history – merge them or insert the assistant reply.
- **Unbounded history** – long chats grow cost and can exceed the context window; trim or summarise.

## Key Takeaways
- The user prompt is the per-turn request in the `user` role.
- Conversations are lists of alternating user/assistant messages you resend each call.
- Treat user content as untrusted and delimit it.
- Templates turn raw input + context into a well-structured user prompt.

## Related Questions
- [What is a system prompt?](10-What-is-a-system-prompt.md)
- [What is prompt engineering?](09-What-is-prompt-engineering.md)
- [What is the difference between a list and a tuple?](../Week-05-Python-for-LLM-Engineering/03-What-is-the-difference-between-a-list-and-a-tuple.md)
- [How do you manage API costs and rate limits?](17-How-do-you-manage-API-costs-and-rate-limits.md)
