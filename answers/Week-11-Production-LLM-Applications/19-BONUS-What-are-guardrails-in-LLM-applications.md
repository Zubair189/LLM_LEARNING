# What are guardrails in LLM applications?

> Week 11 – Production LLM Applications · Bonus question (added)

**BONUS** – This question was added to tie together the safety topics of Week 11 (prompt injection, hallucination detection, SQL restriction and human approval).

## In One Line
Guardrails are **checks in your code around the model** that stop bad inputs going in, bad answers coming out, and dangerous actions happening.

## Simple Explanation
The model can be tricked or make mistakes, and its own instructions can be bypassed. So we add safety checks **outside** the model, in normal code. There are three kinds:

1. **Input guardrails** (before the model):
   - Is the message too long?
   - Does it look like a prompt injection?
   - Does it contain private data, like a card number, that we should hide first?
   - Is it on-topic for this app?
2. **Output guardrails** (after the model):
   - Is the answer in the right format (e.g. valid JSON)?
   - Is it supported by the sources (no hallucination)?
   - Does it leak private data or secrets?
3. **Action guardrails** (when the model uses tools):
   - Only allowed tools, read-only database access, human approval for risky actions.

```
User → [Input checks] → LLM → [Action checks] → [Output checks] → User
                                                     └─ fail → block / retry / ask a human
```

**Start simple:** plain Python rules (length limits, patterns, JSON checks) are cheap and fast. Add smarter checks, like a small classifier model or an LLM judge, only where needed, because each check adds time and cost.

## Real-Life Example
Highway guardrails don't steer your car. They stop it going off a cliff when something goes wrong. LLM guardrails don't write the answer; they catch the dangerous ones.

## Code Example
One input guardrail (size limit, hide card numbers) and one output guardrail (valid JSON, no card numbers).

```python
import json, re

CARD = re.compile(r"\b\d(?:[ -]?\d){12,15}\b")       # looks like a card number

def input_guard(text):
    if len(text) > 2000:
        raise ValueError("input too long")            # rule 1: size limit
    return CARD.sub("[CARD HIDDEN]", text)            # rule 2: hide private data

def output_guard(raw):
    try:
        reply = json.loads(raw)                       # rule 3: must be valid JSON
    except json.JSONDecodeError:
        return "Sorry, I couldn't give a safe answer."
    if CARD.search(reply.get("answer", "")):
        return "Sorry, I couldn't give a safe answer."  # rule 4: no card numbers out
    return reply["answer"]

print(input_guard("My card 4111 1111 1111 1111 was charged twice"))
print(output_guard('{"answer": "Contact your bank to dispute it."}'))
print(output_guard("Here is some text that is not JSON"))
# My card [CARD HIDDEN] was charged twice
# Contact your bank to dispute it.
# Sorry, I couldn't give a safe answer.
```

**What happened:** the card number was hidden before the model could see it, a good JSON answer passed, and a badly formatted answer was replaced with a safe message.

## Watch Out
- The system prompt is **not** a guardrail. Real guardrails live in code outside the model.
- Too many strict checks can block harmless users. Measure how often you block, and tune.

## Remember
- Guardrails = **input checks + output checks + action limits**, in code.
- Start with cheap rules; add smarter checks where needed.
- Decide what happens on failure: block, retry or ask a human.

## Related Questions
- [What is prompt injection?](06-What-is-prompt-injection.md)
- [What is hallucination detection?](14-What-is-hallucination-detection.md)
- [Why should AI-generated SQL be restricted?](17-Why-should-AI-generated-SQL-be-restricted.md)
- [How do you secure an LLM application?](../Week-12-Final-Project-and-Revision/10-How-do-you-secure-an-LLM-application.md)
