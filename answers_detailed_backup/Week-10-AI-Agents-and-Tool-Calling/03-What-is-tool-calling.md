# What is tool calling?

> Week 10 – AI Agents and Tool Calling · Question 3

## Short Answer
Tool calling is the way an LLM asks your application to run an external capability on its behalf. You describe the available tools (name, description, JSON-schema inputs). The model replies with a structured request to call one of them, your code executes it, and you send the result back. This lets the model use live data and take actions instead of only producing text.

## Detailed Explanation
**The protocol (Anthropic Messages API):**
1. You send `tools=[...]` along with the messages.
2. The model returns `stop_reason == "tool_use"` and one or more `tool_use` content blocks, each with `id`, `name` and `input`.
3. Your code runs the tool.
4. You send back a `user` message containing `tool_result` blocks that reference the `tool_use_id`.
5. The model continues. It may call more tools, or it finishes with `end_turn`.

```
You ──tools + question──▶ Model
You ◀─tool_use{name,input,id}── Model
You ──run tool locally──▶ result
You ──tool_result{tool_use_id}──▶ Model ──▶ final answer
```

**What counts as a "tool":**
- Client tools that you implement (your API or database).
- Server tools run by the provider (web search, code execution).
- Tools exposed through **MCP** servers, a standard that lets any compliant client discover and call them.

**Tool calling vs. function calling.** The two terms mean essentially the same thing. "Function calling" was OpenAI's original 2023 name. "Tool calling" is the more general term used today by Anthropic, OpenAI's newer APIs, and MCP. It also covers built-in or server-side tools that are not your own functions. See [Q4](04-What-is-function-calling.md).

## Example
**Analogy:** a manager (the LLM) fills in a work-order form ("Tool: weather, City: Paris"). A technician (your code) does the job and attaches the report. The manager never touches the equipment.

```python
import json, anthropic
client = anthropic.Anthropic()

tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city. Call when the user asks about weather or what to wear.",
    "input_schema": {"type": "object",
                     "properties": {"city": {"type": "string", "description": "City name, e.g. 'Paris'"}},
                     "required": ["city"]},
}]
messages = [{"role": "user", "content": "Weather in Paris?"}]
resp = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                              tools=tools, messages=messages)

block = next(b for b in resp.content if b.type == "tool_use")
print(block.name, block.input)          # -> get_weather {'city': 'Paris'}

result = json.dumps({"temp_c": 18, "conditions": "light rain"})  # your real API call
messages += [
    {"role": "assistant", "content": resp.content},
    {"role": "user", "content": [{"type": "tool_result",
                                  "tool_use_id": block.id, "content": result}]},
]
final = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                               tools=tools, messages=messages)
print(final.content[0].text)            # -> "It's 18°C with light rain in Paris."
```

## Common Mistakes / Misconceptions
- **Thinking the model executes the tool.** It only emits a request. Execution, and therefore security, is your job.
- **Writing vague descriptions.** The model picks tools largely from their name and description.
- **Dropping the assistant `tool_use` turn** before sending `tool_result`. The API rejects orphaned results.
- **Trusting `block.input` blindly.** Validate it like any untrusted user input.

## Key Takeaways
- Tool calling means structured requests from the model, executed by your code.
- Tools are defined with a name, a description and a JSON schema.
- Results go back as `tool_result` blocks keyed by `tool_use_id`.
- MCP standardises how tools are exposed, so they can be reused across apps.

## Related Questions
- [What is function calling?](04-What-is-function-calling.md)
- [How does an LLM decide which tool to use?](05-How-does-an-LLM-decide-which-tool-to-use.md)
- [What happens when a tool call fails?](12-What-happens-when-a-tool-call-fails.md)
- [What is the Model Context Protocol (MCP)?](16-BONUS-What-is-the-Model-Context-Protocol-MCP.md)
