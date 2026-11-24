# What is function calling?

> Week 10 – AI Agents and Tool Calling · Question 4

## Short Answer
Function calling is the ability of an LLM to output a structured, schema-valid request to call a function you defined, such as `get_weather(city="Paris")`, instead of plain text. In practice it is **largely a synonym for tool calling**. "Function calling" is the older, OpenAI-originated name. "Tool calling" is the broader modern term, and it also covers server-side and MCP tools.

## Detailed Explanation
**Same mechanism, different names:**

| Term | Where you see it | Nuance |
|---|---|---|
| Function calling | OpenAI (2023 `functions` param), Gemini, many tutorials | Emphasises mapping to *your code's functions* |
| Tool calling / tool use | Anthropic (`tools`, `tool_use`), OpenAI's newer `tools`, MCP | Broader: functions **plus** built-in/server tools (web search, code execution) and remote MCP tools |

The useful nuances:
- **Function calling as structured output.** Some teams use a "function" schema without ever running anything, just to get reliable JSON (for example, extracting `{name, email}` from text). The model fills the schema, and you read the arguments.
- **Tool calling as action.** Usually implies execution and a loop that feeds results back (agents).
- **Who executes.** With client tools/functions, *your* code runs them. With server tools, the provider runs them. With MCP tools, an MCP server runs them.

In interviews, say: *"They're the same core capability. Function calling is the original name. Tool calling is the general term that includes non-function tools and MCP."*

```
           ┌──────────── "tool calling" ────────────┐
           │  ┌── "function calling" ──┐             │
           │  │ your Python/JS funcs   │  server     │
           │  └────────────────────────┘  tools, MCP │
           └────────────────────────────────────────┘
```

## Example
**Analogy:** "soda" vs. "soft drink". The meaning is the same. One word is regional or historical, and the other is the umbrella term.

Using a tool definition purely for **structured extraction**, which is the classic "function calling" use:

```python
import anthropic
client = anthropic.Anthropic()

tools = [{
    "name": "record_contact",
    "description": "Record a contact extracted from the text.",
    "input_schema": {"type": "object",
                     "properties": {"name": {"type": "string"},
                                    "email": {"type": "string"}},
                     "required": ["name", "email"]},
}]
resp = client.messages.create(
    model="claude-sonnet-5-5", max_tokens=1024, tools=tools,
    tool_choice={"type": "tool", "name": "record_contact"},   # force this tool
    messages=[{"role": "user",
               "content": "Hi, I'm Ana Silva, reach me at ana@example.com"}])

args = next(b.input for b in resp.content if b.type == "tool_use")
print(args)   # -> {'name': 'Ana Silva', 'email': 'ana@example.com'}
```

No function was executed. The schema simply made the output reliably structured. For the execute-and-loop version, see [Q3](03-What-is-tool-calling.md) and [Q11](11-What-is-an-agent-execution-loop.md).

## Common Mistakes / Misconceptions
- **Treating the two terms as different technologies.** They are the same core capability.
- **Assuming the model calls your function directly.** It produces arguments, and your code decides whether to call.
- **Skipping argument validation** because "it matches the schema". Schemas constrain shape, not intent or safety.
- **Thinking forced tool choice is free.** Forcing a tool means the model can't decline when the tool doesn't fit.

## Key Takeaways
- Function calling ≈ tool calling. Use whichever term your provider uses.
- "Tool calling" is the umbrella term: functions, server tools and MCP tools.
- Function schemas are also a strong technique for structured outputs.
- Execution and permissions always stay in your application.

## Related Questions
- [What is tool calling?](03-What-is-tool-calling.md)
- [How does an LLM decide which tool to use?](05-How-does-an-LLM-decide-which-tool-to-use.md)
- [What is the Model Context Protocol (MCP)?](16-BONUS-What-is-the-Model-Context-Protocol-MCP.md)
