# How do you maintain conversation history?

> Week 11 – Production LLM Applications · Question 4

## In One Line
The model forgets everything between calls, so **your app** saves the chat and sends the recent messages back with every new question.

## Simple Explanation
LLM APIs are **stateless**. That means each call is brand new: the model does not remember what you said one minute ago.

So how do chatbots "remember"? The app does the remembering:

1. **Save** every user message and every model reply for this chat **session** (one conversation).
2. When a new question arrives, **load** the saved messages.
3. **Send** the recent messages plus the new question to the model.
4. **Save** the new question and answer, ready for next time.

Where to save it:
- A Python dict: fine for a demo, but it is lost when the app restarts.
- **Redis** (a fast shared store) with an expiry time: a common choice in real apps.
- A database: when you need to keep chats long-term.

**Why not send everything forever?** The model has a **context window** (a limit on how much text it can read at once), and you pay for every token you send. So you keep it small:
- **Keep the last few messages** (a "sliding window").
- **Summarise** older messages into a short note.

## Real-Life Example
Imagine calling a help line where each new agent has no memory. The company keeps a case file. Before answering, each agent reads the last few notes in the file. The file is the conversation history.

## Code Example
Keep only the last 4 past messages and send them with each new question.

```python
KEEP = 4                             # keep the last 4 past messages (2 question/answer pairs)
history = []                         # the app's memory for one chat session

def fake_llm(messages):              # stand-in for the real model call
    return f"(reply after reading {len(messages)} messages)"

def chat(user_text):
    new_msg = {"role": "user", "content": user_text}
    messages = history[-KEEP:] + [new_msg]       # recent history + the new question
    reply = fake_llm(messages)                   # the model sees only `messages`
    history.append(new_msg)                      # save the question...
    history.append({"role": "assistant", "content": reply})   # ...and the answer
    return reply

for text in ["Hi, I'm Ana", "I like tea", "What's my name?", "Thanks!"]:
    print(chat(text))
print("Stored messages:", len(history))
# Output:
# (reply after reading 1 messages)
# (reply after reading 3 messages)
# (reply after reading 5 messages)
# (reply after reading 5 messages)
# Stored messages: 8
```

**What happened:** the app stored all 8 messages, but never sent more than 5 to the model. That keeps cost and size under control.

## Watch Out
- The model does **not** remember on its own. If you don't send it, the model never sees it.
- When trimming, keep whole question/answer pairs so the list still starts with a `user` message.

## Remember
- Memory lives in **your app**, not in the model.
- Save per session, send back a **trimmed** or summarised history.
- Longer history = more tokens = more cost.

## Related Questions
- [What is context management?](05-What-is-context-management.md)
- [What is token consumption?](11-What-is-token-consumption.md)
- [What is short-term memory?](../Week-10-AI-Agents-and-Tool-Calling/07-What-is-short-term-memory.md)
- [What is a context window?](../Week-02-Introduction-to-LLMs-and-Generative-AI/09-What-is-a-context-window.md)
