# Explain AI agent architecture.

> Week 12 – Final Project and Revision · Question 8

## Short Answer
An AI agent wraps an LLM in a **loop** that lets it pursue a goal: the LLM (the "brain") **reasons** about the next step, **chooses a tool**, the runtime **executes** it, the **observation** is fed back, and the loop repeats until the goal is met or a limit is hit. The core components are the **LLM**, **tools** (APIs, search, code, DB), **memory** (short-term conversation/scratchpad and long-term stores), **planning/control logic**, and **guardrails** such as permissions, step limits, and human approval for risky actions.

## Detailed Explanation

### End-to-end architecture

```
                     ┌────────────────────────────────────────┐
  User goal ────────▶│            Orchestrator / runtime       │
  "Refund order 123  │  ┌──────────────────────────────────┐   │
   if eligible"      │  │ LLM (reason + decide next action)│   │
                     │  └───────┬───────────────▲──────────┘   │
                     │   tool_use│               │tool_result   │
                     │          ▼               │               │
                     │  ┌────────────────┐  ┌───┴───────────┐   │
                     │  │ Policy/guards  │─▶│ Tool executor │   │
                     │  │ allowlist,     │  │ get_order()   │   │
                     │  │ args validate, │  │ check_policy()│   │
                     │  │ human approval │  │ issue_refund()│   │
                     │  └────────────────┘  └───────────────┘   │
                     │  Memory: short-term (messages, scratch)  │
                     │          long-term (vector DB, profile)  │
                     │  Limits: max steps, timeout, budget      │
                     └───────────────────┬──────────────────────┘
                                         ▼
                      Final answer + audit log / traces
```

### Components

| Component | Role | Production concern |
|---|---|---|
| LLM | Decides which action and arguments | Model choice, cost per step |
| Tools | Do real work: search, DB read, send email | Least privilege, input validation |
| Execution loop | Calls LLM → runs tool → appends result → repeat | Max iterations, timeouts |
| Short-term memory | Messages + tool results in the current task | Context window growth |
| Long-term memory | Facts across sessions (vector DB, key-value) | Privacy, staleness |
| Planner | Break goal into steps (implicit or explicit plan) | Over-planning, loops |
| Guardrails / HITL | Block or pause sensitive actions | Approval UX, audit trail |
| Observability | Trace each step, tool call, tokens | Debugging multi-step failures |

### Patterns
- **Single-agent tool loop (ReAct-style):** simplest and most common.
- **Workflow (fixed steps with LLM nodes):** predictable; prefer this when the path is known.
- **Multi-agent:** e.g., planner + researcher + writer; useful for complex tasks but adds cost and failure modes.

### LLM vs agent
A plain LLM call maps input → text once. An agent **acts**, observes results, and adapts over multiple steps. That power is also the risk: errors compound, costs grow per step, and actions have side effects.

## Example
**Analogy:** A travel agent (LLM) with a phone (tools), a notepad (short-term memory), your customer file (long-term memory), and a manager who must sign off on purchases over $500 (human-in-the-loop).

**Python – minimal agent loop with Claude tool use and an approval gate:**
```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
TOOLS = [
    {"name": "get_order", "description": "Look up an order by id",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}},
                      "required": ["order_id"]}},
    {"name": "issue_refund", "description": "Refund an order (sensitive)",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}},
                      "required": ["order_id"]}},
]
SENSITIVE = {"issue_refund"}

def run_tool(name, args):
    if name == "get_order":
        return {"order_id": args["order_id"], "days_since_purchase": 10, "total": 40}
    if name == "issue_refund":
        return {"status": "refunded"}

def agent(goal: str, max_steps: int = 5) -> str:
    messages = [{"role": "user", "content": goal}]
    for _ in range(max_steps):
        resp = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
            system="You are a support agent. Refunds allowed within 30 days.",
            tools=TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":
            return resp.content[0].text
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                if block.name in SENSITIVE and input(f"Approve {block.name}{block.input}? y/n ") != "y":
                    out = {"error": "denied by human reviewer"}
                else:
                    out = run_tool(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(out)})
        messages.append({"role": "user", "content": results})
    return "Stopped: step limit reached."

print(agent("Refund order 123 if eligible."))
# Approve issue_refund{'order_id': '123'}? y/n y
# Order 123 was purchased 10 days ago, so it's eligible; the refund has been issued.
```

## Common Mistakes / Misconceptions
- **No step/budget limit** — agents can loop and burn tokens.
- **Giving tools broad permissions** (e.g., write SQL, delete files) — enforce least privilege in code, not in the prompt.
- **Using an agent when a fixed workflow would do** — more autonomy = less predictability.
- **Trusting tool outputs blindly** — web pages and documents can carry prompt injection.

## Key Takeaways
- Agent = LLM + tools + memory + loop + guardrails.
- The runtime, not the LLM, executes tools — that's where you enforce permissions.
- Bound every loop: max steps, timeouts, token budget.
- Use human approval for irreversible or sensitive actions.
- Trace every step for debugging and audit.

## Related Questions
- [What is an AI agent?](../Week-10-AI-Agents-and-Tool-Calling/01-What-is-an-AI-agent.md)
- [What is the difference between an LLM and an AI agent?](../Week-10-AI-Agents-and-Tool-Calling/02-What-is-the-difference-between-an-LLM-and-an-AI-agent.md)
- [What is an agent execution loop?](../Week-10-AI-Agents-and-Tool-Calling/11-What-is-an-agent-execution-loop.md)
- [What is agent memory?](../Week-10-AI-Agents-and-Tool-Calling/06-What-is-agent-memory.md)
- [What is a multi-agent system?](../Week-10-AI-Agents-and-Tool-Calling/10-What-is-a-multi-agent-system.md)
- [What are the limitations of autonomous agents?](../Week-10-AI-Agents-and-Tool-Calling/15-What-are-the-limitations-of-autonomous-agents.md)
- [How do you implement human approval for sensitive operations?](../Week-11-Production-LLM-Applications/18-How-do-you-implement-human-approval-for-sensitive-operations.md)
- [How does function calling work?](09-How-does-function-calling-work.md)
