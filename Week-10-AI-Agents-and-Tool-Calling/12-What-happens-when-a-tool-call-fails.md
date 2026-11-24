# What happens when a tool call fails?

> Week 10 – AI Agents and Tool Calling · Question 12

## Short Answer
That depends on how you handle it. In a well-built agent, the failure is caught and returned to the model as a `tool_result` with `"is_error": True` and a clear message. The model can then retry with corrected arguments, try a different tool, ask the user, or explain that it couldn't finish. In a badly built agent, the exception crashes the loop, or the error is silently swallowed and the model hallucinates a result.

## Detailed Explanation
**Types of failure:**

| Failure | Example | Best handling |
|---|---|---|
| Invalid arguments | Missing field, wrong type, unknown city | Return a descriptive error; the model fixes and retries |
| Transient | Timeout, 429 rate limit, 503 | Retry in *code* with exponential backoff, then report the error |
| Permanent | 404, auth denied, tool not allowlisted | Return the error; don't retry blindly |
| Unknown tool name | Model invents `get_wether` | Return an "unknown tool; available: ..." error |
| Bad/partial output | Empty result, huge payload | Validate and trim; say "no results" explicitly |
| Side-effect uncertainty | Payment timed out: did it go through? | Idempotency keys; check status before retrying |

**Handling flow:**
```
tool_use ─▶ validate args ──fail──▶ tool_result(is_error, "city is required")
              │ok
              ▼
           execute ──transient──▶ retry ×3 w/ backoff ──still fail──▶ is_error
              │ok                                                   ▲
              ▼                                                     │
           tool_result(content)              permanent error ───────┘
```

**Guardrails:**
- Write error messages that are actionable for the model: say what went wrong *and* how to fix it.
- Cap retries per tool and set a total iteration limit. Otherwise the model can retry the same failure forever.
- Never leak secrets or stack traces in error text.
- Escalate to a human after repeated failures on important tasks.

## Example
**Analogy:** you ask a colleague to book a room and it's taken. A good colleague comes back with "Room A is booked, B is free at 3pm, shall I book that?" A bad one says nothing, or pretends it worked.

```python
import time

class TransientError(Exception): ...

def run_tool(block, registry, retries=3):
    """Return a tool_result dict; never raise."""
    fn = registry.get(block.name)
    if fn is None:
        return {"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                "content": f"Unknown tool '{block.name}'. Available: {list(registry)}"}
    for attempt in range(retries):
        try:
            return {"type": "tool_result", "tool_use_id": block.id,
                    "content": str(fn(**block.input))}
        except TransientError:
            time.sleep(2 ** attempt)                     # 1s, 2s, 4s backoff
        except (TypeError, ValueError) as e:             # bad args: let model fix
            return {"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                    "content": f"Invalid input: {e}. Check the schema and try again."}
    return {"type": "tool_result", "tool_use_id": block.id, "is_error": True,
            "content": "Service unavailable after 3 attempts. Tell the user to try later."}

# Inside the loop from Q11:
# results = [run_tool(b, REGISTRY) for b in resp.content if b.type == "tool_use"]
# messages.append({"role": "user", "content": results})
# Typical model follow-up: "Sorry, I couldn't find 'Pariss' — did you mean Paris?"
```

## Common Mistakes / Misconceptions
- **Letting exceptions escape the loop.** The whole task dies, and so does the conversation state.
- **Returning an empty string on error.** The model may treat it as success and make up data.
- **Retrying non-idempotent actions blindly**, which can lead to double payments or duplicate emails.
- **Unbounded self-correction loops.** The model keeps trying the same failing call.

## Key Takeaways
- Treat errors as information for the model: `tool_result` + `is_error: True`.
- Retry transient errors in code with backoff. Return permanent errors immediately.
- Make error messages specific and actionable, without leaking secrets.
- Bound retries, use idempotency for side effects, and escalate to humans when stuck.

## Related Questions
- [What is an agent execution loop?](11-What-is-an-agent-execution-loop.md)
- [What is tool calling?](03-What-is-tool-calling.md)
- [What are human-in-the-loop systems?](13-What-are-human-in-the-loop-systems.md)
- [What are the limitations of autonomous agents?](15-What-are-the-limitations-of-autonomous-agents.md)
