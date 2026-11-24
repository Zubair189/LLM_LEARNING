# What is an agent execution loop?

> Week 10 – AI Agents and Tool Calling · Question 11

## Short Answer
The agent execution loop is the core control cycle of an agent. Call the model; if it asks for tools, run them and append the results; call the model again; repeat until it produces a final answer or a limit is hit. It is often described as **reason → act → observe** (the ReAct pattern). With the Anthropic SDK, you loop while `resp.stop_reason == "tool_use"`.

## Detailed Explanation
```
 messages = [user goal]
 ┌─────────────────────────────────────────────┐
 │ resp = model(messages, tools)               │
 │ if resp.stop_reason != "tool_use": break ───┼──▶ final answer
 │ append assistant turn (resp.content)        │
 │ for each tool_use block: run → tool_result  │
 │ append user turn with all tool_results      │
 │ iterations += 1; if > MAX: stop safely      │
 └─────────────────────────────────────────────┘
```

**Essential ingredients:**
1. **Append the assistant's full `resp.content`.** It contains the `tool_use` blocks that the results refer to.
2. **Return *all* tool results in one user message**, each with the matching `tool_use_id`. The model may request several tools in parallel.
3. **Handle errors as data.** Return `is_error: True` with a message so the model can recover (see [Q12](12-What-happens-when-a-tool-call-fails.md)).
4. **Stop conditions.** `end_turn`, a max-iteration cap, a token/cost budget, wall-clock timeouts, and `max_tokens` truncation.
5. **Guardrails inside the loop.** A tool allowlist (a dispatch dict, never `eval`), argument validation, and human approval for risky tools.
6. **Observability.** Log every step: the tool, its inputs, latency and errors.

SDKs also provide ready-made runners (for example the Anthropic SDK tool runner, or the Claude Agent SDK). You should still understand the manual loop.

## Example
**Analogy:** a cook following a recipe they're inventing: taste, adjust, taste again. They stop when it's good, or when the timer (max iterations) goes off.

```python
import json, anthropic
client = anthropic.Anthropic()

tools = [{"name": "get_weather",
          "description": "Current weather for a city.",
          "input_schema": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}]

def get_weather(city: str) -> str:
    data = {"Paris": "18°C rain", "Rome": "26°C sunny"}
    if city not in data:
        raise ValueError(f"Unknown city: {city}")
    return data[city]

REGISTRY = {"get_weather": get_weather}      # allowlist
MAX_ITERS = 8

messages = [{"role": "user", "content": "Compare weather in Paris and Rome."}]
for i in range(MAX_ITERS):
    resp = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                                  tools=tools, messages=messages)
    if resp.stop_reason != "tool_use":
        break
    messages.append({"role": "assistant", "content": resp.content})
    results = []
    for block in resp.content:
        if block.type != "tool_use":
            continue
        try:
            out = REGISTRY[block.name](**block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        except Exception as e:
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": f"Error: {e}", "is_error": True})
    messages.append({"role": "user", "content": results})
else:
    print("Stopped: max iterations reached")

print(resp.content[0].text)
# -> "Rome is warmer and sunny (26°C), while Paris is 18°C with rain."
```

## Common Mistakes / Misconceptions
- **No iteration cap**, so the agent can loop forever and burn budget.
- **Sending tool results without appending the assistant turn first.** The API returns an error.
- **Handling only the first `tool_use` block** when the model requested several.
- **Raising exceptions out of the loop** instead of returning `is_error` results the model can react to.

## Key Takeaways
- Loop: model → tool_use? → execute → tool_result → model … → end_turn.
- Keep the message structure exact: an assistant turn, then a user turn with `tool_result`s.
- Always bound the loop: iterations, tokens, time.
- Allowlist, validate and log every tool execution.

## Related Questions
- [What is tool calling?](03-What-is-tool-calling.md)
- [What happens when a tool call fails?](12-What-happens-when-a-tool-call-fails.md)
- [How do you prevent an agent from performing unauthorized operations?](14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
- [What is an agent workflow?](09-What-is-an-agent-workflow.md)
