# What is a system prompt?

> Week 06 – Working with LLMs and Model APIs · Question 10

> Note: The **system prompt** is the developer's standing instructions for the whole chat, while the **user prompt** ([Q11](11-What-is-a-user-prompt.md)) is the request for one turn.

## In One Line
A system prompt is the **set of background instructions** the developer gives the model, telling it how to behave for the whole conversation.

## Simple Explanation
When you build an app with an LLM, you want it to behave the same way every time. You put those standing rules in the system prompt.

It usually contains:

1. **Role:** "You are a support bot for Acme."
2. **Rules:** "Only answer questions about Acme products."
3. **Style:** "Be short and friendly. Reply in the user's language."
4. **Format:** "Use bullet points" or "Reply in JSON".
5. **Background facts:** company info, today's date, and so on.

The **user** never sees or types the system prompt. The **developer** writes it once, and it applies to every message in the chat.

Where it goes:
- **Anthropic API:** a separate `system=` setting.
- **Ollama / OpenAI-style APIs:** a message with `"role": "system"`.

Models are trained to follow the system prompt closely, but it is **guidance, not a lock**. A tricky user may still try to talk the model out of it, so important rules should also be checked in your code.

**Link to LLMs:** the system prompt is the main way you turn a general model into *your* specific assistant.

## Real-Life Example
The system prompt is like the job description and handbook a new employee gets on day one. The customer requests they handle afterwards are the user prompts.

## Code Example
```python
import anthropic

client = anthropic.Anthropic()                            # reads ANTHROPIC_API_KEY
SYSTEM = "You are a pirate-themed coding tutor. Answer in under 40 words and end with 'Arrr!'."

resp = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=1024,
    system=SYSTEM,                                         # the system prompt (its own setting)
    messages=[{"role": "user", "content": "What is a Python list?"}],  # the user prompt
)
print(resp.content[0].text)

# Output (e.g.):
# A list be an ordered, changeable collection of items, like [1, 2, 3]. Arrr!
```

**What happened:** the user only asked about lists, but the answer followed the system prompt's style, length and ending.

## Watch Out
- With the Anthropic API, don't put `"role": "system"` inside `messages`. Use the `system=` setting.
- Don't put secrets (like API keys) in the system prompt. Users can sometimes get the model to reveal it.

## Remember
- System prompt = **developer's standing instructions**.
- Use it for **role, rules, style and format**.
- Strong guidance, but **not a security wall**.

## Related Questions
- [What is a user prompt?](11-What-is-a-user-prompt.md)
- [What is prompt engineering?](09-What-is-prompt-engineering.md)
- [What is few-shot prompting?](18-BONUS-What-is-few-shot-prompting.md)
- [What is a dictionary?](../Week-05-Python-for-LLM-Engineering/04-What-is-a-dictionary.md)
