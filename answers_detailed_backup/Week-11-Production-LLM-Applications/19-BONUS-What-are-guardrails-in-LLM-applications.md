# What are guardrails in LLM applications?

> Week 11 – Production LLM Applications · Bonus question (added)

**BONUS** – This question was added to tie together the safety topics of Week 11 (prompt injection, hallucination detection, SQL restriction and human approval).

## Short Answer
Guardrails are the checks and controls placed *around* an LLM so that it behaves within defined limits. **Input guardrails** run before the model call (validation, injection/abuse detection, PII redaction, topic filters). **Output guardrails** run after it (schema validation, groundedness checks, toxicity/PII filters, policy rules). **Action guardrails** constrain what tools can do (allowlists, read-only access, human approval). They are implemented in code, often with help from smaller classifier models, because the model's own instructions can be bypassed.

## Detailed Explanation
```
             ┌──────────── INPUT GUARDRAILS ────────────┐
user input ─►│ size/format · injection & jailbreak check │
             │ PII redaction · topic/allowed-use filter  │
             └───────────────────┬───────────────────────┘
                                 ▼
                         LLM (+ system prompt)
                                 │
             ┌──────────── ACTION GUARDRAILS ───────────┐
             │ tool allowlist · read-only DB · approvals │
             └───────────────────┬───────────────────────┘
                                 ▼
             ┌──────────── OUTPUT GUARDRAILS ───────────┐
             │ JSON/schema · grounding/citations · PII   │
             │ toxicity · secrets · link/URL allowlist   │
             └───────────────────┬───────────────────────┘
                                 ▼
                pass ─► user      fail ─► block / retry / fallback / human
```

| Guardrail type | Examples | Implementation |
|---|---|---|
| Deterministic rules | Max length, regex for card numbers, JSON schema, SQL parser | Plain code, Pydantic, `sqlglot` |
| Classifier-based | Injection detector, toxicity, topic relevance | Small model or cheap LLM call (`claude-haiku-5-5`) |
| LLM-as-judge | Groundedness, policy compliance | Separate judge prompt |
| Structural | Least-privilege tools, human approval, rate limits | Architecture and permissions |

**Trade-offs:** every guardrail adds latency and cost and can produce false positives that block legitimate users. Put cheap deterministic checks first, run expensive checks only when needed, and measure block rates. Frameworks such as Guardrails AI, NeMo Guardrails and Llama Guard-style classifiers help, but simple Python checks cover most needs.

## Example
**Analogy:** Highway guardrails don't steer your car. They stop it from going off a cliff when something goes wrong. LLM guardrails don't write the answer; they catch the dangerous ones.

```python
import re
import anthropic
from pydantic import BaseModel, ValidationError

client = anthropic.Anthropic()
CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")

class Reply(BaseModel):
    answer: str
    confidence: float

def input_guard(text: str) -> str:
    if len(text) > 2000:
        raise ValueError("input too long")
    return CARD.sub("[REDACTED_CARD]", text)          # PII redaction

def output_guard(raw: str) -> Reply:
    reply = Reply.model_validate_json(raw)            # schema check
    if CARD.search(reply.answer):
        raise ValueError("PII in output")
    return reply

def ask(question: str) -> Reply | str:
    q = input_guard(question)
    r = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=300,
        system='Reply ONLY with JSON: {"answer": str, "confidence": number 0-1}.',
        messages=[{"role": "user", "content": q}])
    try:
        return output_guard(r.content[0].text)
    except (ValidationError, ValueError):
        return "Sorry, I couldn't produce a safe answer."

print(ask("My card 4111 1111 1111 1111 was charged twice, what do I do?"))
# -> answer='Contact your bank and dispute the duplicate charge...' confidence=0.8  (illustrative)
# The model never saw the card number: it was redacted by the input guardrail.
```

## Common Mistakes / Misconceptions
- Treating the system prompt as the guardrail. Guardrails must live in code outside the model.
- Only guarding inputs or only outputs. You need both, plus action limits.
- Stacking so many LLM-based checks that latency and cost triple.
- Never measuring false positives, so users get blocked for harmless requests.

## Key Takeaways
- Guardrails = input checks + output checks + action limits, enforced in code.
- Start with cheap deterministic rules; add classifiers or judges where needed.
- Decide the fail action: block, retry, fall back or escalate to a human.
- Monitor block rates and tune; guardrails are part of evaluation and observability.

## Related Questions
- [What is prompt injection?](06-What-is-prompt-injection.md)
- [What is hallucination detection?](14-What-is-hallucination-detection.md)
- [Why should AI-generated SQL be restricted?](17-Why-should-AI-generated-SQL-be-restricted.md)
- [How do you secure an LLM application?](../Week-12-Final-Project-and-Revision/10-How-do-you-secure-an-LLM-application.md)
