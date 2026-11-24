# What is a multi-agent system?

> Week 10 – AI Agents and Tool Calling · Question 10

## Short Answer
A multi-agent system uses several LLM agents, each with its own role, prompt, tools and context, that cooperate on a task. A common setup is an **orchestrator** (lead agent) that breaks the work into pieces and delegates them to **sub-agents** (workers), often in parallel, then combines their results. It helps with broad, parallelizable tasks and keeps each agent's context focused. The costs are more tokens, more latency and harder debugging.

## Detailed Explanation
**Common topologies:**
```
 Orchestrator–workers          Pipeline / handoff         Peer debate / review
      [Lead]                   [Research]─▶[Write]─▶[Edit]   [Agent A] ⇄ [Agent B]
     /   |   \                                                    \     /
 [W1]  [W2]  [W3]                                                 [Judge]
```

**Why use multiple agents?**
- **Context isolation.** Each sub-agent explores in its own context window and returns a short summary, so the lead's context stays clean.
- **Parallelism.** Independent sub-tasks run at the same time, for example researching 5 companies.
- **Specialisation.** Different prompts, tools or models per role (a cheap model for search, a strong model for synthesis).
- **Least privilege per role.** The "reader" agent gets read-only tools, and only the "executor" can write.

**Costs and risks:**

| Issue | Detail |
|---|---|
| Tokens | Multi-agent runs can use several times (even 10×+) the tokens of a single agent |
| Coordination | Duplicate work, conflicting outputs, lost information in handoffs |
| Debugging | You need tracing across agents |
| Error propagation | One agent's hallucination becomes another's "fact" |

**Rule of thumb.** Try a single agent with good tools first. Go multi-agent when tasks are clearly parallel or one context window can't hold the exploration.

**Interop.** In 2026, agents share tools through **MCP**. Agent-to-agent protocols (such as A2A) are emerging for cross-vendor agent communication.

## Example
**Analogy:** a newsroom. The editor (orchestrator) assigns stories to reporters (sub-agents), each reporter researches separately, and the editor merges their pieces into one edition.

```python
import anthropic
from concurrent.futures import ThreadPoolExecutor

client = anthropic.Anthropic()

def agent(role: str, task: str, max_tokens: int = 512) -> str:
    r = client.messages.create(model="claude-sonnet-5-5", max_tokens=max_tokens,
                               system=f"You are a {role}. Be concise.",
                               messages=[{"role": "user", "content": task}])
    return r.content[0].text

topic = "Adopting LoRA fine-tuning at a startup"
subtasks = {"cost analyst": f"Estimate GPU costs for: {topic}",
            "risk analyst": f"List top 3 risks of: {topic}",
            "ML engineer":  f"Outline a 5-step plan for: {topic}"}

with ThreadPoolExecutor() as pool:                      # workers in parallel
    results = dict(zip(subtasks, pool.map(lambda kv: agent(*kv), subtasks.items())))

report = agent("lead editor", "Merge into a one-page brief:\n\n" +
               "\n\n".join(f"## {k}\n{v}" for k, v in results.items()), max_tokens=1024)
print(report[:80])   # -> "## Brief: Adopting LoRA at a Startup ..."
```

## Common Mistakes / Misconceptions
- **"More agents = smarter."** Often it only means more cost and more coordination bugs.
- **Vague delegation.** Sub-agents need clear objectives, an output format and boundaries.
- **Sharing one giant context across agents**, which defeats the isolation benefit.
- **Giving every agent every tool.** Scope tools per role.

## Key Takeaways
- Multi-agent = several role-specialised agents, usually an orchestrator plus workers.
- Main benefits: parallelism and context isolation.
- Main costs: tokens, latency, coordination and debuggability.
- Start single-agent. Scale out only when the task demands it.

## Related Questions
- [What is an agent workflow?](09-What-is-an-agent-workflow.md)
- [What is an AI agent?](01-What-is-an-AI-agent.md)
- [What are the limitations of autonomous agents?](15-What-are-the-limitations-of-autonomous-agents.md)
- [What is the Model Context Protocol (MCP)?](16-BONUS-What-is-the-Model-Context-Protocol-MCP.md)
