# What is an agent execution loop?

> Week 10 – AI Agents and Tool Calling · Question 11

## In One Line
The execution loop is the heart of an agent: **ask the model → run the tools it asks for → give back the results → ask again**, until it gives a final answer.

## Simple Explanation
The loop is plain code that you write. Each round goes like this:

1. **Call the model** with the conversation so far and the list of tools.
2. **Check its reply.**
   - If it asks to use tools, go to step 3.
   - If it gives a normal answer, you're **done**.
3. **Run each requested tool** in your code.
4. **Add the results** to the conversation. Each result is tagged with the ID of the request it answers.
5. **Go back to step 1.**

People often describe this as **reason → act → observe** (sometimes called the **ReAct** pattern).

With the Anthropic API, the model's reply has a `stop_reason`. If it is `"tool_use"`, the model wants a tool, so you keep looping. Anything else (like `"end_turn"`) means it's finished.

**Always add a safety limit**, such as a maximum of 10 rounds. Otherwise a confused model could loop forever and run up a big bill.

## Real-Life Example
It's like a cook following a recipe with a helper. The cook says "pass me the salt"; the helper passes it; the cook tastes and says "now the pepper"; and so on, until the cook says "dinner's ready". The helper also has a rule: if it takes more than an hour, stop and ask what's wrong.

## Code Example
The loop with the Anthropic Python SDK (needs an API key, so it is not run here).

```python
import anthropic
client = anthropic.Anthropic()

tools = [{"name": "get_weather", "description": "Get the weather for a city.",
          "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}]
messages = [{"role": "user", "content": "Weather in Paris and Rome?"}]

for _ in range(10):                                        # safety limit
    resp = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                                  tools=tools, messages=messages)
    if resp.stop_reason != "tool_use":                     # no tool needed -> done
        break
    messages.append({"role": "assistant", "content": resp.content})  # keep its request
    results = [{"type": "tool_result", "tool_use_id": b.id,          # answer each request
                "content": f"{b.input['city']}: 18°C"}               # fake tool result
               for b in resp.content if b.type == "tool_use"]
    messages.append({"role": "user", "content": results})            # send results back

print(resp.content[0].text)   # e.g. "Both Paris and Rome are 18°C right now."
```

**What happened:** while the model kept asking for tools, we ran them and sent back results tagged with `tool_use_id`. When it stopped asking, we printed its final answer.

## Watch Out
- Forgetting to add the model's own reply (`resp.content`) before the tool results breaks the conversation.
- With no round limit, a bug can mean an endless, expensive loop.

## Remember
- Loop: call model → run tools → return results → repeat.
- Keep looping while `stop_reason == "tool_use"`.
- Always set a maximum number of rounds.

## Related Questions
- [What is tool calling?](03-What-is-tool-calling.md)
- [What happens when a tool call fails?](12-What-happens-when-a-tool-call-fails.md)
- [How do you prevent an agent from performing unauthorized operations?](14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
- [What is an agent workflow?](09-What-is-an-agent-workflow.md)
