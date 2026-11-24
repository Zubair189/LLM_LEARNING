# What is a user prompt?

> Week 06 – Working with LLMs and Model APIs · Question 11

> Note: The [system prompt](10-What-is-a-system-prompt.md) is the developer's standing instructions, while the user prompt is the specific request for this turn.

## In One Line
A user prompt is the **actual question or task** sent to the model in one turn of the conversation.

## Simple Explanation
A chat with an LLM is stored as a **list of messages**. Each message has a **role**:

- **`user`**: what the person (or your code) asks. This is the user prompt.
- **`assistant`**: what the model replied.

They take turns: user, assistant, user, assistant...

**The model has no memory between calls.** If the user asks "And its population?", the model only knows what "its" means if you send the earlier messages too. So your app keeps the whole list and sends it every time.

Sometimes your code builds the user prompt from a **template**, for example:
"Use only these documents: ... Question: ..."

**Be careful with user text.** It comes from outside, so it can't be fully trusted. Someone might type "ignore your rules". Put user text inside tags like `<question>...</question>` and keep important rules in your code.

**Link to LLMs:** the `messages=[...]` list in every API call is made of user and assistant turns.

## Real-Life Example
If the system prompt is a restaurant's menu and kitchen rules, the user prompt is each customer's order. The kitchen cooks each order, but always within its rules.

## Code Example
How an app keeps the conversation (fake replies so it runs without an API).

```python
history = []                                        # the whole conversation

def add_turn(user_text, fake_reply):
    history.append({"role": "user", "content": user_text})        # the user prompt
    history.append({"role": "assistant", "content": fake_reply})  # the model's answer

add_turn("What's the capital of France?", "Paris.")
add_turn("And its population?", "About 2 million in the city.")

for msg in history:                                 # this list is sent on every call
    print(msg["role"], "->", msg["content"])
print(len(history), "messages")

# Output:
# user -> What's the capital of France?
# assistant -> Paris.
# user -> And its population?
# assistant -> About 2 million in the city.
# 4 messages
```

**What happened:** each turn added a user message and an assistant reply. In a real app you pass `messages=history` to `client.messages.create(...)`, so the model can understand that "its" means Paris.

## Watch Out
- Sending only the newest message makes the model "forget" everything before it.
- Very long histories cost more tokens and can go over the model's limit. Trim or summarise old turns.

## Remember
- User prompt = **the request for this turn**.
- Chats are lists of **user / assistant** messages, sent **every** time.
- Treat user text as **untrusted**.

## Related Questions
- [What is a system prompt?](10-What-is-a-system-prompt.md)
- [What is prompt engineering?](09-What-is-prompt-engineering.md)
- [What is the difference between a list and a tuple?](../Week-05-Python-for-LLM-Engineering/03-What-is-the-difference-between-a-list-and-a-tuple.md)
- [How do you manage API costs and rate limits?](17-How-do-you-manage-API-costs-and-rate-limits.md)
