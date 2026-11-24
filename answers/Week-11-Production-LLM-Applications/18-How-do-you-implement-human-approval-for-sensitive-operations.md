# How do you implement human approval for sensitive operations?

> Week 11 – Production LLM Applications · Question 18

## In One Line
The LLM may **suggest** a risky action, like a refund, but your code **parks it and waits** until a real person approves it.

## Simple Explanation
Some actions are hard to undo: sending money, deleting data, emailing customers. We don't want a model, which can make mistakes or be tricked, doing these alone. This pattern is called **human-in-the-loop**.

**Step 1: sort actions by risk.**
- **Low risk** (just reading, like "check order status"): run straight away.
- **High risk** (refunds, deletes, payments, sending emails): need a human "yes".

**Step 2: the approval flow.**
1. The model asks to call a tool, e.g. `issue_refund(order=123, amount=250)`.
2. Your code sees it is high risk and **saves it as "pending"** instead of running it.
3. A person (e.g. a manager) is shown the **exact details**: which order, how much.
4. They approve or reject.
5. Only on approval does **your code** run the action, and it writes down who approved what (an **audit log**).

```
LLM suggests action → Risky? ─no→ run it
                         └─yes→ save as pending → human reviews → approve → run + log
                                                               └→ reject → tell user
```

**Good rules:**
- The **code** does the action, never the model directly.
- Approve the **exact details**. If anything changes, approve again.
- Each approval can be used **only once**, and old requests should expire.
- Don't ask for approval on everything, or people start clicking "yes" without reading.

## Real-Life Example
A junior bank clerk can fill in the form for a big money transfer, but it doesn't leave the bank until a manager reads the form and signs it.

## Code Example
Low-risk actions run at once; high-risk ones wait for a person.

```python
HIGH_RISK = {"issue_refund", "delete_customer"}   # actions a human must approve
pending = {}                                      # actions waiting for a decision

def propose(action_id, tool, args):               # the LLM asks to do something
    if tool not in HIGH_RISK:
        return f"done: {tool}{args}"              # low risk -> run straight away
    pending[action_id] = (tool, args)             # high risk -> park it and wait
    return f"waiting for approval: {tool}{args}"

def decide(action_id, approver, approve):         # a human looks at the exact details
    tool, args = pending.pop(action_id)           # pop: each request is used only once
    print(f"audit log: {approver} {'approved' if approve else 'rejected'} {tool}{args}")
    return f"done: {tool}{args}" if approve else "not done: rejected by a human"

print(propose("a1", "get_order_status", {"order": 123}))
print(propose("a2", "issue_refund", {"order": 123, "amount": 250}))
print(decide("a2", approver="maria", approve=True))
# done: get_order_status{'order': 123}
# waiting for approval: issue_refund{'order': 123, 'amount': 250}
# audit log: maria approved issue_refund{'order': 123, 'amount': 250}
# done: issue_refund{'order': 123, 'amount': 250}
```

**What happened:** reading the order ran immediately. The refund waited until Maria approved it, and her decision was logged.

## Watch Out
- Writing "always ask the user first" in the prompt is **not enough**. The model can skip it. Enforce approval in code.
- Show the approver the **exact** amount and target, not a vague summary.

## Remember
- The LLM **suggests**, the code **enforces**, a human **approves**.
- Only high-risk actions need approval.
- Keep an **audit log** of who approved what.

## Related Questions
- [What are human-in-the-loop systems?](../Week-10-AI-Agents-and-Tool-Calling/13-What-are-human-in-the-loop-systems.md)
- [How do you prevent an agent from performing unauthorized operations?](../Week-10-AI-Agents-and-Tool-Calling/14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
- [Why should AI-generated SQL be restricted?](17-Why-should-AI-generated-SQL-be-restricted.md)
- [What is an indirect prompt injection attack?](07-What-is-an-indirect-prompt-injection-attack.md)
