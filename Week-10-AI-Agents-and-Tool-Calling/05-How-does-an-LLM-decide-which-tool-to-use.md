# How does an LLM decide which tool to use?

> Week 10 – AI Agents and Tool Calling · Question 5

## Short Answer
The LLM chooses a tool the same way it generates any text: by predicting the most likely next tokens given its context. That context includes the user request, the conversation so far, the system prompt, and each tool's **name, description and input schema**. Models are fine-tuned to emit a structured tool call when a tool's description matches the need. Clear, distinct tool descriptions are therefore the main lever you control.

## Detailed Explanation
**What the model "sees":** tool definitions are serialized into the prompt (in a special format) ahead of the conversation. Every tool costs input tokens and attention.

**Signals that drive the choice:**
1. **Description match.** "Get current weather for a city" vs. the question "Will it rain?".
2. **Name.** `search_orders` is clearer than `tool2`.
3. **Schema.** Required parameters it can fill from context make a tool more likely to be chosen.
4. **System prompt guidance.** For example: "Use `kb_search` before answering policy questions."
5. **Prior results.** After a failed or empty result, the model may switch tools.
6. **`tool_choice`.** `auto` (the model decides), `any` (must call some tool), `{"type":"tool","name":...}` (forced), or `none`.

**Training.** Models learn this behaviour through fine-tuning on tool-use traces and preference tuning (see [Week 09 bonus](../Week-09-Fine-tuning-and-Model-Optimization/16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md)). It is not symbolic routing. The choice is probabilistic and can be wrong.

**Scaling to many tools:**

| # tools | Typical issue | Mitigation |
|---|---|---|
| 1–10 | Fine | Good descriptions |
| 10–50 | Confusion between similar tools | Merge overlapping tools, namespacing (`github_create_issue`) |
| 50+ (e.g. many MCP servers) | Token cost, wrong picks | Tool search or dynamic loading, router step, expose only allowlisted tools per task |

## Example
**Analogy:** a new employee picks a department from the company directory. Good entries ("Billing — refunds, invoices") lead to correct routing. Vague entries ("Team B") cause misroutes.

```python
import anthropic
client = anthropic.Anthropic()

tools = [
  {"name": "get_weather",
   "description": "Current weather (temp, conditions) for a city. Use for weather/clothing questions.",
   "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
  {"name": "get_time",
   "description": "Current local time in a city. Use for 'what time is it' questions.",
   "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
]
for q in ["Do I need a coat in Oslo?", "Is it late in Sydney right now?", "Tell me a joke"]:
    r = client.messages.create(model="claude-sonnet-5-5", max_tokens=300, tools=tools,
                               messages=[{"role": "user", "content": q}])
    picked = [b.name for b in r.content if b.type == "tool_use"]
    print(q, "->", picked or r.stop_reason)
# Do I need a coat in Oslo? -> ['get_weather']
# Is it late in Sydney right now? -> ['get_time']
# Tell me a joke -> end_turn        (no tool needed)
```

## Common Mistakes / Misconceptions
- **Believing the model "executes logic" to pick a tool.** It is learned pattern-matching over descriptions.
- **Writing one-word descriptions.** Say what the tool does, when to use it, and when not to.
- **Exposing every available tool**, such as all tools from five MCP servers. This hurts accuracy and cost, and it violates least privilege.
- **Overlapping tools** (`search_docs` vs. `find_documents`) with no guidance on which to use.

## Key Takeaways
- Tool choice is next-token prediction conditioned on tool names, descriptions and schemas.
- Descriptions are prompts. Write them carefully and test them.
- Use `tool_choice` to force or forbid tool use when needed.
- Fewer, well-separated tools give better choices. Scope tools per task.

## Related Questions
- [What is tool calling?](03-What-is-tool-calling.md)
- [What is function calling?](04-What-is-function-calling.md)
- [What happens when a tool call fails?](12-What-happens-when-a-tool-call-fails.md)
- [What is the Model Context Protocol (MCP)?](16-BONUS-What-is-the-Model-Context-Protocol-MCP.md)
