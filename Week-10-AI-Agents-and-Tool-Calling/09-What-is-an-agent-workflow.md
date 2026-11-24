# What is an agent workflow?

> Week 10 – AI Agents and Tool Calling · Question 9

## Short Answer
An agent workflow is a structured sequence of LLM calls, tool calls and logic that accomplishes a task. The term is often used in contrast to a fully autonomous agent. In a **workflow** the *developer* defines the control flow: steps, branches and order. In an **agent** the *model* decides the next step dynamically. Most production systems are workflows with small agentic parts, because workflows are more predictable, testable and cheaper.

## Detailed Explanation
**The autonomy spectrum:**
```
 single LLM call ─▶ fixed workflow ─▶ workflow w/ agentic steps ─▶ autonomous agent
   predictable                                                      flexible
   cheap, testable                                                  costly, harder to test
```

**Common workflow patterns:**

| Pattern | How it works | Example |
|---|---|---|
| Prompt chaining | Output of step 1 feeds step 2 (with checks between) | Outline → draft → proofread |
| Routing | A classifier sends input to a specialised path | Billing vs. tech-support prompts |
| Parallelization | Run sub-tasks concurrently, then aggregate; or vote | Review code for security, style and bugs in parallel |
| Orchestrator–workers | An LLM splits the task and delegates to sub-calls | Multi-file code change |
| Evaluator–optimizer | Generate → critique → revise loop | Translation polishing |
| Agent loop | Model picks tools until done | Open-ended research |

**Choosing:**
- Steps known in advance → **workflow** (deterministic code, plus LLM calls).
- Steps unknown or variable → **agent**. Still bound it with max iterations, tool allowlists and human approval for risky actions.

**Gates between steps.** Validate outputs programmatically (schema checks, tests, regex) before moving on. This is what makes workflows reliable.

## Example
**Analogy:** a workflow is a recipe: chop, then sauté, then simmer. An agent is a chef told "make dinner from what's in the fridge", who decides the steps on the fly.

```python
import anthropic
client = anthropic.Anthropic()

def llm(prompt: str) -> str:
    r = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                               messages=[{"role": "user", "content": prompt}])
    return r.content[0].text

def support_workflow(ticket: str) -> str:
    # Step 1: route (developer-defined branch)
    category = llm(f"Classify as BILLING or TECH. Reply with one word.\n\n{ticket}").strip().upper()
    # Step 2: specialised prompt
    if category == "BILLING":
        draft = llm(f"As a billing agent, draft a polite reply:\n{ticket}")
    else:
        draft = llm(f"As a tech-support agent, give troubleshooting steps:\n{ticket}")
    # Step 3: gate / check
    if len(draft) > 2000:
        draft = llm(f"Shorten to under 150 words:\n{draft}")
    return f"[{category}] {draft}"

print(support_workflow("I was charged twice this month."))
# -> "[BILLING] Sorry about the double charge... we've opened a refund request..."
```

## Common Mistakes / Misconceptions
- **Using a fully autonomous agent for a task with known steps.** A simple pipeline would be cheaper and more reliable.
- **Running no checks between steps.** Errors propagate and compound.
- **Thinking workflows aren't "real AI".** Most successful LLM products are well-designed workflows.
- **Building elaborate frameworks too early.** Start with plain code and a few LLM calls.

## Key Takeaways
- Workflow = developer-controlled flow. Agent = model-controlled flow.
- Learn the core patterns: chaining, routing, parallelization, orchestrator–workers, evaluator–optimizer.
- Add programmatic gates between steps.
- Use the least autonomy that works, and add agentic steps only where flexibility pays off.

## Related Questions
- [What is an agent execution loop?](11-What-is-an-agent-execution-loop.md)
- [What is a multi-agent system?](10-What-is-a-multi-agent-system.md)
- [What are human-in-the-loop systems?](13-What-are-human-in-the-loop-systems.md)
- [What is an AI agent?](01-What-is-an-AI-agent.md)
