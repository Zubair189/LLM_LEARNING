# How does function calling work?

> Week 12 – Final Project and Revision · Question 9

## Short Answer
Function calling (tool use) lets an LLM ask *your code* to run a function. You send the model a list of tools, each with a **name, description, and JSON Schema** for its inputs. If the model decides a tool is needed, it returns a structured **tool-use request** (tool name + JSON arguments) instead of plain text; **your application validates and executes** the function and sends the result back as a **tool result**, and the model uses it to produce the final answer. The model never executes anything itself.

## Detailed Explanation

### End-to-end sequence

```
 App                                   LLM API                          Your function
  │ 1. messages + tools[{name, desc,      │                                   │
  │    input_schema}]  ─────────────────▶ │                                   │
  │                                       │ 2. decides a tool is needed       │
  │ ◀──── stop_reason="tool_use"          │                                   │
  │       {name:"get_weather",            │                                   │
  │        input:{"city":"Paris"}, id}    │                                   │
  │ 3. validate args (Pydantic), check    │                                   │
  │    permissions ──────────────────────────────────────────────────────────▶│
  │ ◀───────────────────────────────────────────── {"temp_c": 18, "sky":"sun"}│
  │ 4. append assistant tool_use +        │                                   │
  │    user tool_result(id) ────────────▶ │                                   │
  │                                       │ 5. writes final answer            │
  │ ◀──── stop_reason="end_turn"          │                                   │
  │       "It's 18°C and sunny in Paris." │                                   │
```

### How the model "chooses"
The model was trained to emit structured tool calls. It reads the tool **descriptions** and schemas in the prompt and picks the one that fits the user's request, filling arguments from the conversation. So good tool descriptions are effectively prompt engineering.

### Key pieces
| Piece | Purpose |
|---|---|
| `tools` list | Name, description, JSON Schema for inputs |
| `tool_use` block | Model's request: `id`, `name`, `input` |
| `tool_result` block | Your output, linked by `tool_use_id` |
| `stop_reason` | `"tool_use"` → run tools; `"end_turn"` → done |
| `tool_choice` | Force a specific tool, any tool, or let the model decide |

### Function calling vs structured output
Forcing a single tool with a schema is also a reliable way to get **structured JSON** (e.g., extract fields from an invoice) even if you never "call" anything.

### Safety rules
- The model's arguments are **untrusted input** — validate types, ranges, and permissions.
- Use **allowlists** of tools; keep each tool narrow (`get_order(id)` not `run_sql(query)`).
- Require **human approval** for sensitive tools; log every call.
- Handle tool **errors** by returning an error result so the model can recover or explain.

## Example
**Analogy:** A manager (LLM) fills out a request form ("Please fetch the Paris weather report") and hands it to an assistant (your code). The assistant checks the form is valid and allowed, does the task, and returns the report; the manager then writes the reply. The manager never touches the filing cabinet directly.

**Python – one round trip with the Anthropic SDK:**
```python
import json
import anthropic
from pydantic import BaseModel

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

class WeatherArgs(BaseModel):
    city: str

def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 18, "sky": "sunny"}   # stub

tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city. Use when the user asks about weather.",
    "input_schema": {"type": "object",
                     "properties": {"city": {"type": "string"}},
                     "required": ["city"]},
}]
messages = [{"role": "user", "content": "What's the weather in Paris?"}]

resp = client.messages.create(model="claude-sonnet-5-5", max_tokens=512,
                              tools=tools, messages=messages)
if resp.stop_reason == "tool_use":
    call = next(b for b in resp.content if b.type == "tool_use")
    args = WeatherArgs(**call.input)                    # validate model output
    result = get_weather(args.city)
    messages += [
        {"role": "assistant", "content": resp.content},
        {"role": "user", "content": [{"type": "tool_result",
                                      "tool_use_id": call.id,
                                      "content": json.dumps(result)}]},
    ]
    resp = client.messages.create(model="claude-sonnet-5-5", max_tokens=512,
                                  tools=tools, messages=messages)
print(resp.content[0].text)
# It's currently 18°C and sunny in Paris.
```

## Common Mistakes / Misconceptions
- **"The LLM runs the function."** It only proposes a call; your code executes it.
- **Vague tool descriptions** → wrong tool or missing arguments.
- **Not validating arguments** — the model can produce wrong types or malicious values (e.g., via prompt injection).
- **Forgetting to send the `tool_result` with the matching `tool_use_id`** — the conversation breaks.

## Key Takeaways
- You describe tools with JSON Schema; the model returns structured calls; you execute and return results.
- `stop_reason == "tool_use"` drives the loop; agents are this loop repeated.
- Tool descriptions guide selection; keep tools narrow and well-named.
- Treat arguments as untrusted; enforce permissions and approvals in code.

## Related Questions
- [What is function calling?](../Week-10-AI-Agents-and-Tool-Calling/04-What-is-function-calling.md)
- [What is tool calling?](../Week-10-AI-Agents-and-Tool-Calling/03-What-is-tool-calling.md)
- [How does an LLM decide which tool to use?](../Week-10-AI-Agents-and-Tool-Calling/05-How-does-an-LLM-decide-which-tool-to-use.md)
- [What happens when a tool call fails?](../Week-10-AI-Agents-and-Tool-Calling/12-What-happens-when-a-tool-call-fails.md)
- [What is structured output?](../Week-06-Working-with-LLMs-and-Model-APIs/15-What-is-structured-output.md)
- [What is JSON?](../Week-05-Python-for-LLM-Engineering/09-What-is-JSON.md)
- [How do you prevent an agent from performing unauthorized operations?](../Week-10-AI-Agents-and-Tool-Calling/14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
- [Explain AI agent architecture.](08-Explain-AI-agent-architecture.md)
