# How do you implement human approval for sensitive operations?

> Week 11 – Production LLM Applications · Question 18

## Short Answer
Classify every tool or action by risk. Let the LLM *propose* sensitive actions (refunds, emails, deletions, payments, data writes) but not *execute* them. The proposal is stored as a pending request with its exact parameters and shown to an authorised human, who approves or rejects it. Only then does the backend execute it, re-validating permissions and logging who approved what. This is the human-in-the-loop (HITL) pattern applied as a hard control in code, not as a request in the prompt.

## Detailed Explanation
**Risk tiers**
| Tier | Examples | Policy |
|---|---|---|
| Low (read-only) | Search docs, read order status | Auto-execute |
| Medium | Draft an email, create a ticket | Auto-execute with logging, or user confirm |
| High | Send email externally, issue refund > $X, delete data, run write SQL, payments | **Mandatory human approval** |

**Flow**
```
User ─► LLM decides to call tool "issue_refund(order=123, amount=250)"
              │
              ▼
   Backend policy check: tool is HIGH risk
              │
              ▼
   Save PendingAction{id, tool, args, requested_by, reason, status=PENDING}
              │
              ▼
   Notify approver (UI / Slack / email) ──► Approver reviews exact args
              │                                   │
        rejected ◄────────────────────────────────┤ approved
              │                                   ▼
   tell user "not approved"          Backend re-checks authz, idempotency key,
                                     executes, logs audit record, informs user
```

**Design rules**
- **The executor is your code, not the model.** The LLM only produces a structured request.
- **Approve exact parameters.** If anything changes after approval, approval is required again.
- **Separation of duties:** the requester can't approve their own action; approvers need the right role.
- **Expiry & idempotency:** pending actions time out; execute at most once.
- **Audit trail:** who requested, who approved, when, and what was executed.
- Show approvers the model's reasoning and the source data, but they decide on the concrete parameters.

## Example
**Analogy:** A junior bank clerk can prepare a large wire transfer, but it doesn't leave the bank until a manager reviews the form and signs it.

```python
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
HIGH_RISK = {"issue_refund", "delete_customer"}
pending: dict[str, dict] = {}     # use a DB table in production

class ToolCall(BaseModel):
    tool: str
    args: dict
    requested_by: str

def execute(tool: str, args: dict) -> str:
    return f"executed {tool} {args}"          # real side effect goes here

@app.post("/tool-calls")
async def propose(call: ToolCall):
    if call.tool not in HIGH_RISK:
        return {"status": "done", "result": execute(call.tool, call.args)}
    aid = str(uuid.uuid4())
    pending[aid] = {**call.model_dump(), "status": "PENDING",
                    "created": datetime.now(timezone.utc).isoformat()}
    return {"status": "awaiting_approval", "approval_id": aid}

@app.post("/approvals/{aid}")
async def decide(aid: str, approver: str, approve: bool):
    action = pending.get(aid)
    if not action or action["status"] != "PENDING":
        raise HTTPException(404, "no pending action")
    if approver == action["requested_by"]:
        raise HTTPException(403, "cannot approve own request")
    action.update(status="APPROVED" if approve else "REJECTED", approver=approver)
    if approve:
        action["result"] = execute(action["tool"], action["args"])
    print({"audit": aid, "status": action["status"], "approver": approver})
    return action
# POST /tool-calls {"tool":"issue_refund","args":{"order":123,"amount":250},"requested_by":"bot"}
# -> {"status":"awaiting_approval","approval_id":"6f1c..."}
# POST /approvals/6f1c...?approver=maria&approve=true
# -> {"status":"APPROVED","result":"executed issue_refund {'order': 123, 'amount': 250}", ...}
```

## Common Mistakes / Misconceptions
- Asking the model in the prompt to "confirm with the user first". The model can skip that. Enforce it in code.
- Showing approvers a vague summary instead of the exact parameters that will run.
- Approval fatigue: requiring approval for everything, so people click "approve" without reading. Tier by risk.
- No audit log or expiry, so old approvals can be replayed.

## Key Takeaways
- The LLM proposes; code enforces; humans approve high-risk actions.
- Tier tools by risk and require approval only where it matters.
- Approve exact parameters, enforce separation of duties, expiry and idempotency.
- Keep a complete audit trail.

## Related Questions
- [What are human-in-the-loop systems?](../Week-10-AI-Agents-and-Tool-Calling/13-What-are-human-in-the-loop-systems.md)
- [How do you prevent an agent from performing unauthorized operations?](../Week-10-AI-Agents-and-Tool-Calling/14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
- [Why should AI-generated SQL be restricted?](17-Why-should-AI-generated-SQL-be-restricted.md)
- [What is an indirect prompt injection attack?](07-What-is-an-indirect-prompt-injection-attack.md)
