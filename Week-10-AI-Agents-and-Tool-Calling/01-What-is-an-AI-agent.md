# What is an AI agent?

> Week 10 – AI Agents and Tool Calling · Question 1

## Short Answer
An AI agent is a system where an LLM works in a loop toward a goal. It decides what to do next, calls tools (APIs, code, search, files) to act on the world, looks at the results, and keeps going until the task is done or a stop condition is reached. A chatbot answers one message. An agent plans, acts, observes and adapts across several steps.

## Detailed Explanation
A practical definition: **agent = LLM + tools + loop + memory + guardrails**.

| Component | Role |
|---|---|
| LLM ("brain") | Reasons about the goal, picks the next action |
| Tools | Functions the model can request: search, DB query, send email, run code. Often exposed through **MCP** servers |
| Loop | Call model → run tool → feed result back → repeat |
| Memory | Short-term (the conversation/context) and long-term (vector store, files, DB) |
| Guardrails | Max iterations, allowlisted tools, least-privilege credentials, sandboxing, human approval for risky actions |

```
 Goal ─▶ [LLM thinks] ─▶ tool_use? ──yes──▶ [run tool] ─▶ result ─┐
              ▲                                                  │
              └──────────────────────────────────────────────────┘
                         │ no (end_turn)
                         ▼
                    Final answer
```

**Autonomy is a spectrum.**
- *Workflow*: you hard-code the steps, and the LLM fills in pieces.
- *Tool-using assistant*: one or two tool calls per user turn.
- *Autonomous agent*: the model chooses the steps and the number of iterations.

In 2026 most production systems sit in the middle. They give the agent bounded autonomy, and a human approves irreversible actions.

## Example
**Analogy:** a chatbot is like a librarian who answers from memory. An agent is like a research assistant. It goes to the shelves, makes phone calls, checks the results, and comes back with a finished report.

```python
import anthropic

client = anthropic.Anthropic()
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city. Use when the user asks about weather.",
    "input_schema": {"type": "object",
                     "properties": {"city": {"type": "string"}},
                     "required": ["city"]},
}]

def get_weather(city: str) -> str:
    return f"{city}: 18°C, light rain"   # stub for a real API

messages = [{"role": "user", "content": "Should I bring an umbrella in Paris?"}]
for _ in range(5):                                   # max-iteration guard
    resp = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                                  tools=tools, messages=messages)
    if resp.stop_reason != "tool_use":
        break
    messages.append({"role": "assistant", "content": resp.content})
    results = []
    for block in resp.content:
        if block.type == "tool_use":
            out = get_weather(**block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
    messages.append({"role": "user", "content": results})

print(resp.content[0].text)
# e.g. "Yes — it's 18°C with light rain in Paris, so bring an umbrella."
```

## Common Mistakes / Misconceptions
- **Assuming "agent" means fully autonomous.** Most useful agents are tightly scoped and supervised.
- **Thinking the LLM runs the tools itself.** The model only *requests* a call. Your code runs it.
- **Building an agent when a fixed workflow would do.** Agents add cost, latency and unpredictability.
- **Shipping without limits.** An agent with no iteration cap or permission boundary can loop forever or do damage.

## Key Takeaways
- An agent is an LLM in a perceive → decide → act → observe loop.
- Tools give it the ability to act. MCP is the common standard for plugging tools in.
- Memory and guardrails matter as much as the model does.
- Pick the least autonomy that solves the problem.

## Related Questions
- [What is the difference between an LLM and an AI agent?](02-What-is-the-difference-between-an-LLM-and-an-AI-agent.md)
- [What is an agent execution loop?](11-What-is-an-agent-execution-loop.md)
- [What is tool calling?](03-What-is-tool-calling.md)
- [What is the Model Context Protocol (MCP)?](16-BONUS-What-is-the-Model-Context-Protocol-MCP.md)
