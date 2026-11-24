# What is a multi-agent system?

> Week 10 – AI Agents and Tool Calling · Question 10

## In One Line
A multi-agent system is a **team of AI agents**, each with its own job, working together on one bigger task.

## Simple Explanation
Instead of one agent doing everything, you split the work between several agents. Each one has its own **role**, **instructions**, **tools** and **context** (what it sees).

The most common setup is a **lead agent with helpers**:

1. The **orchestrator** (lead agent) receives the big task and splits it into smaller pieces.
2. It hands each piece to a **sub-agent** (a worker), often running at the same time.
3. Each sub-agent works on its piece and reports back a short summary.
4. The orchestrator combines the results into the final answer.

**Why do this?**
- **Speed:** sub-agents can work in parallel.
- **Focus:** each agent sees only what it needs, so its context stays small and clean.
- **Specialists:** a "researcher" and a "writer" can have different tools and instructions.

**The costs:**
- **More tokens, more money:** many agents means many LLM calls.
- **Harder to debug:** a mistake can hide anywhere in the team.
- **Coordination problems:** agents may repeat work or misunderstand each other.

Good fit: broad tasks that split into independent parts, such as researching ten companies. Poor fit: tightly connected tasks where every step depends on the last, such as editing one small piece of code.

## Real-Life Example
A newspaper editor (orchestrator) gets the task "cover the election". They send one reporter to each city (sub-agents). Each reporter writes a short piece, and the editor combines them into one front page.

## Code Example
A toy orchestrator that splits a task among workers and combines their results.

```python
def research_worker(company):               # a sub-agent (fake, no AI needed)
    return f"{company}: growing fast"

def orchestrator(task, companies):
    print("Lead agent plans:", task)
    results = [research_worker(c) for c in companies]   # hand out the pieces
    return " | ".join(results)                         # combine the reports

print(orchestrator("Compare companies", ["Acme", "Globex", "Initech"]))

# Lead agent plans: Compare companies
# Acme: growing fast | Globex: growing fast | Initech: growing fast
```

**What happened:** the lead agent split the job, each worker handled one company, and the lead combined the answers. In a real system each worker would be its own LLM agent, often running in parallel.

## Watch Out
- More agents is not automatically better. Start with one agent and add more only when it clearly helps.
- Sub-agents should return short summaries, not huge dumps, or the lead agent's context overflows.

## Remember
- Multi-agent = a team of agents with different roles.
- Usual pattern: orchestrator + sub-agents.
- Faster and more focused, but costlier and harder to debug.

## Related Questions
- [What is an agent workflow?](09-What-is-an-agent-workflow.md)
- [What is an AI agent?](01-What-is-an-AI-agent.md)
- [What are the limitations of autonomous agents?](15-What-are-the-limitations-of-autonomous-agents.md)
- [What is the Model Context Protocol (MCP)?](16-BONUS-What-is-the-Model-Context-Protocol-MCP.md)
