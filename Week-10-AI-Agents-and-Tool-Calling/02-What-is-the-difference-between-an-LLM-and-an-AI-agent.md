# What is the difference between an LLM and an AI agent?

> Week 10 – AI Agents and Tool Calling · Question 2

## Short Answer
An LLM is a model: it takes text in and returns text out, in a single stateless call. An AI agent is a *system built around* an LLM. It adds tools, a control loop, memory and guardrails, so it can take actions over several steps toward a goal. The LLM is the engine. The agent is the whole car.

## Detailed Explanation

| Aspect | LLM (bare model call) | AI agent |
|---|---|---|
| Unit of work | One request → one response | A goal → many steps |
| State | Stateless; it only sees what you send | Keeps conversation and task state, often long-term memory |
| Actions | Produces text only | Calls tools: APIs, code, files, browsers, MCP servers |
| Knowledge | Frozen at training cutoff (plus the prompt) | Can fetch fresh data through tools or retrieval |
| Control flow | None | A loop decides when to continue and when to stop |
| Failure surface | Wrong text | Wrong *actions*, so it needs permissions, sandboxing and approvals |
| Cost/latency | One call | N calls; tokens grow with every step |

**Key nuance.** Modern LLM APIs support tool calling natively. The model can *output* a structured request such as "call `get_weather` with `{"city":"Paris"}`". That alone does not make it an agent. It only becomes an agent when **your code** runs the tool, sends the result back, and repeats.

```
LLM:    prompt ─▶ model ─▶ text

Agent:  goal ─▶ ┌──────── loop (max N) ────────┐
                │ model ─▶ tool request         │
                │   ▲          │                 │
                │   └─ result ◀ executor (yours)│
                └───────────────────────────────┘ ─▶ answer
```

## Example
**Analogy:** an LLM is a brilliant consultant on the phone who can only give advice. An agent is the same consultant with a laptop, access to your systems, a to-do list, and a rule to "check with me before spending money".

```python
import anthropic
client = anthropic.Anthropic()

# Bare LLM: one call, no tools
r = client.messages.create(model="claude-sonnet-5-5", max_tokens=200,
    messages=[{"role": "user", "content": "What's the weather in Tokyo right now?"}])
print(r.content[0].text)
# -> "I don't have access to real-time data..."  (it can't act)

# Agent: same model + tools + loop (see Q11 for the full loop)
tools = [{"name": "get_weather", "description": "Current weather for a city.",
          "input_schema": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}]
r = client.messages.create(model="claude-sonnet-5-5", max_tokens=200, tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Tokyo right now?"}])
print(r.stop_reason)   # -> "tool_use"  (it asks YOUR code to fetch the data)
```

## Common Mistakes / Misconceptions
- **"GPT/Claude is an agent."** The model is not an agent. Products like Claude Code are agents built *on* models.
- **Thinking that adding a tool definition creates an agent.** Without an execution loop, nothing ever runs.
- **Believing agents are always smarter.** They are more capable *and* more error-prone, because mistakes compound across steps.

## Key Takeaways
- An LLM generates text. An agent achieves goals by taking actions.
- An agent is an LLM plus tools, a loop, memory and guardrails.
- Agents need security controls (least privilege, allowlists, sandboxing) that bare LLM calls don't need.
- Start with a single LLM call. Move to an agent only when the task needs several steps or real actions.

## Related Questions
- [What is an AI agent?](01-What-is-an-AI-agent.md)
- [What is an agent execution loop?](11-What-is-an-agent-execution-loop.md)
- [What is agent memory?](06-What-is-agent-memory.md)
- [What are the limitations of autonomous agents?](15-What-are-the-limitations-of-autonomous-agents.md)
