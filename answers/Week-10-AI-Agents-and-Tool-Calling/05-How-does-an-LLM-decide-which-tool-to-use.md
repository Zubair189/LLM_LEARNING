# How does an LLM decide which tool to use?

> Week 10 – AI Agents and Tool Calling · Question 5

## In One Line
The LLM picks a tool the same way it writes any text: it reads the request and the **tool descriptions**, and predicts which tool call fits best.

## Simple Explanation
There is no hidden "if-then" rulebook inside the model. Everything it knows about your tools comes from what you send it:

- the user's request and the conversation so far,
- the system prompt (your instructions),
- each tool's **name**, **description** and **input schema** (the list of inputs it needs).

The model has been specially trained to produce a tool request when a tool's description matches what the user needs. So the decision is really **next-word prediction**, guided by your descriptions.

That means **your tool descriptions are the main thing you control.** Good descriptions:
1. Say clearly **what** the tool does and **when** to use it (and when *not* to).
2. Use clear, distinct names, for example `search_orders` and `refund_order`, not `tool1` and `tool2`.
3. Explain each input with an example, such as "city name, e.g. 'Paris'".
4. Don't overlap. If two tools sound alike, the model gets confused.

You can also guide it in other ways:
- Give fewer tools. Too many tools makes choosing harder.
- Add instructions in the system prompt, such as "always check stock before ordering".
- Many APIs let you force a choice. With Anthropic, `tool_choice` can be `auto` (model decides), `any` (must use some tool), a specific tool, or `none`.

## Real-Life Example
Think of a new employee looking at a wall of labelled buttons. If the labels say "Print invoice" and "Email invoice", they pick correctly. If the labels say "Button A" and "Button B", they guess. The model is the employee; your descriptions are the labels.

## Code Example
A toy chooser that picks the tool whose description shares the most words with the request. **Real LLMs don't count words**; this only shows why clear descriptions help.

```python
tools = {
    "get_weather": "get current weather temperature rain for a city",
    "get_time":    "get current local time for a city",
}

def choose(request):
    words = set(request.lower().split())                      # words in the request
    scores = {name: len(words & set(desc.split()))            # count shared words
              for name, desc in tools.items()}
    return max(scores, key=scores.get)                        # best match wins

print(choose("Will it rain in Paris"))       # get_weather
print(choose("What time is it in Tokyo"))    # get_time
```

**What happened:** each request matched the tool whose description talks about the same thing. A real LLM understands meaning, not just shared words, but it still relies on your descriptions in the same way.

## Watch Out
- If the model picks the wrong tool, fix the **description** first before blaming the model.
- The model can also decide to use **no** tool and just answer. That is often correct.

## Remember
- The model chooses by prediction, using names, descriptions and schemas.
- Clear, distinct descriptions are your main lever.
- Fewer, well-described tools make better choices.

## Related Questions
- [What is tool calling?](03-What-is-tool-calling.md)
- [What is function calling?](04-What-is-function-calling.md)
- [What happens when a tool call fails?](12-What-happens-when-a-tool-call-fails.md)
- [What is the Model Context Protocol (MCP)?](16-BONUS-What-is-the-Model-Context-Protocol-MCP.md)
