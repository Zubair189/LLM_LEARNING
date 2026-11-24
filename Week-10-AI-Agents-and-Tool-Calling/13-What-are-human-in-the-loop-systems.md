# What are human-in-the-loop systems?

> Week 10 – AI Agents and Tool Calling · Question 13

## Short Answer
Human-in-the-loop (HITL) systems put a person at key decision points in an AI workflow. The human approves, edits or rejects actions before they take effect, answers the agent's clarifying questions, or reviews outputs. For agents, HITL is the standard safety control for **high-impact or irreversible actions** such as sending emails, spending money, deleting data or deploying code. It trades some speed for safety, accountability and trust.

## Detailed Explanation
**Where humans plug in:**

| Pattern | When | Example |
|---|---|---|
| Approval gate | Before a risky tool runs | "Agent wants to `send_email` to 2,000 customers. Approve?" |
| Edit/review | Before output is published | Human edits a drafted contract clause |
| Clarification | Agent is uncertain or the request is ambiguous | "Which account: personal or business?" |
| Escalation | Repeated failures, low confidence, policy edge case | Hand off to a support rep |
| Oversight / audit | After the fact, by sampling | Weekly review of 5% of agent transcripts |
| Feedback | Ratings and corrections | Data for evals or preference tuning |

Human-*on*-the-loop is a related idea: the agent acts autonomously while a human monitors it and can intervene.

**Risk-tiered policy (typical):**
```
read-only (search, get_weather)      → auto-run
reversible writes (draft, create PR) → auto-run + log
irreversible / external / $$         → require human approval
forbidden (drop prod DB)             → not exposed at all
```

**Design tips:**
- Show the human *exactly* what will happen: the tool, the arguments, and the diff or preview.
- Keep approvals rare and meaningful. Constant prompts cause "approval fatigue" and rubber-stamping.
- Make rejections informative. Return the reason to the model as a `tool_result` so it can adapt.
- Persist state so the agent can pause and resume after an asynchronous approval (Slack, email, UI).

## Example
**Analogy:** a junior employee can draft any email but needs a manager's sign-off before anything goes to a client or any payment is made.

```python
RISKY = {"send_email", "refund_payment", "delete_record"}

def approve(name: str, args: dict) -> bool:
    ans = input(f"Agent wants to run {name}({args}). Approve? [y/N] ")
    return ans.strip().lower() == "y"

def execute_with_hitl(block, registry):
    if block.name in RISKY and not approve(block.name, block.input):
        return {"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                "content": "Action rejected by human reviewer. Do not retry; "
                           "ask the user how to proceed."}
    out = registry[block.name](**block.input)
    return {"type": "tool_result", "tool_use_id": block.id, "content": str(out)}

# In the agent loop (Q11):
# results = [execute_with_hitl(b, REGISTRY) for b in resp.content if b.type == "tool_use"]
# Console:  Agent wants to run refund_payment({'order': 812, 'amount': 49.0}). Approve? [y/N] n
# Model:    "I've paused the refund — the reviewer didn't approve it. Want me to escalate?"
```

## Common Mistakes / Misconceptions
- **Thinking HITL means a human reviews everything.** That doesn't scale. Gate by risk.
- **Showing the reviewer a vague summary** instead of the actual action and arguments.
- **Treating approval as a substitute for permissions.** Still apply least privilege, because humans miss things.
- **Not telling the model about a rejection.** It will retry or claim success.

## Key Takeaways
- HITL puts human judgment at high-stakes points: approve, edit, clarify, escalate, audit.
- Tier tools by risk: auto-run reads, gate irreversible actions.
- Show precise previews and avoid approval fatigue.
- Combine HITL with least privilege, allowlists and sandboxing for defence in depth.

## Related Questions
- [How do you prevent an agent from performing unauthorized operations?](14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
- [What are the limitations of autonomous agents?](15-What-are-the-limitations-of-autonomous-agents.md)
- [What is an agent workflow?](09-What-is-an-agent-workflow.md)
- [What is preference tuning (RLHF and DPO)?](../Week-09-Fine-tuning-and-Model-Optimization/16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md)
