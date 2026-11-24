# What are human-in-the-loop systems?

> Week 10 – AI Agents and Tool Calling · Question 13

## In One Line
A human-in-the-loop (HITL) system makes the AI **stop and ask a person** before important steps, so a human can approve, change or reject what it is about to do.

## Simple Explanation
Agents can make mistakes or be tricked. For small, safe actions (like reading a file) that's fine. But for **big or hard-to-undo actions** you want a person to check first, such as:
- sending an email to a customer,
- spending money or making a payment,
- deleting data,
- deploying code to a live website.

How it works:
1. The agent decides on an action.
2. Your code checks: **is this a risky action?**
3. If yes, it **pauses** and shows the person exactly what it plans to do.
4. The person **approves**, **edits** or **rejects** it.
5. The agent continues. If rejected, it is told "the user said no" and adjusts its plan.

Other ways humans join the loop:
- **Answering questions:** the agent asks "which account did you mean?"
- **Reviewing output:** a person checks a draft before it's published.
- **Spot checks:** reviewing a sample of the agent's work afterwards.

**The trade-off:** it's slower, but much safer, and someone is clearly accountable. Don't ask about *everything*, though. Too many pop-ups make people click "yes" without reading.

## Real-Life Example
A junior employee can order office supplies alone, but any purchase over $500 needs the manager's signature. The manager doesn't check every pen, only the big, risky spending.

## Code Example
Ask for approval only for risky tools. The human's answer is simulated so this runs without typing.

```python
RISKY = {"send_email", "delete_file", "make_payment"}   # actions that need a human

def ask_human(action, details):
    print(f"APPROVAL NEEDED: {action} {details}")
    return action != "make_payment"     # simulated answer; real code would use input()

def run_action(action, details):
    if action in RISKY and not ask_human(action, details):
        return f"{action}: rejected by human"          # tell the agent it was refused
    return f"{action}: done"

print(run_action("read_file", "notes.txt"))
print(run_action("send_email", "to boss"))
print(run_action("make_payment", "$900"))

# read_file: done
# APPROVAL NEEDED: send_email to boss
# send_email: done
# APPROVAL NEEDED: make_payment $900
# make_payment: rejected by human
```

**What happened:** the safe action ran straight away. The risky ones paused for a person, who approved the email and rejected the payment.

## Watch Out
- The approval check must live in **your code**, not just in the prompt. A prompt can be ignored or tricked.
- Show the person the **exact** action and details, not a vague summary.

## Remember
- HITL = a human approves important steps.
- Use it for costly, risky or irreversible actions.
- Too many approvals cause "just click yes" fatigue.

## Related Questions
- [How do you prevent an agent from performing unauthorized operations?](14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
- [What are the limitations of autonomous agents?](15-What-are-the-limitations-of-autonomous-agents.md)
- [What is an agent workflow?](09-What-is-an-agent-workflow.md)
- [What is preference tuning (RLHF and DPO)?](../Week-09-Fine-tuning-and-Model-Optimization/16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md)
