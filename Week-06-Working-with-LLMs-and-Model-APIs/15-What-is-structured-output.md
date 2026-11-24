# What is structured output?

> Week 06 – Working with LLMs and Model APIs · Question 15

> Note: **Structured output** is the general goal — getting machine-readable data that matches a specific schema. **JSON mode** ([Q16](16-What-is-JSON-mode.md)) is one narrower provider feature that only guarantees *syntactically valid JSON*, not that it has the fields and types you need.

## Short Answer
Structured output means getting an LLM to return data in a predefined, machine-parseable shape — typically JSON matching a schema — rather than free text, so your code can use it directly. Reliable approaches include defining a tool with an `input_schema` and forcing the model to call it, using a provider's native structured-output/JSON-schema mode, and always validating the result with a library like Pydantic before use.

## Detailed Explanation
Free text is great for humans but fragile for programs ("The sentiment is mostly positive, I'd say..."). Structured output turns the LLM into a function: text in → typed object out.

Approaches from least to most reliable:

| Approach | How | Guarantee |
|---|---|---|
| Prompt only | "Reply in JSON with keys name, age" | None — may add prose or wrong keys |
| JSON mode | Provider flag (OpenAI `response_format={"type":"json_object"}`, Ollama `format="json"`) | Valid JSON syntax only |
| Schema-constrained | Tool with `input_schema` + forced `tool_choice`; or native JSON-schema modes (OpenAI `response_format` with `json_schema`, Ollama `format=<schema>`, Anthropic structured outputs) | Matches schema (very high reliability) |
| + Validation | Pydantic/jsonschema check, retry on failure | Your code is protected either way |

```
 prompt + schema ─► LLM ─► JSON ─► Pydantic validate ─► typed object ─► your app
                                       │ fail
                                       └─► retry with error message / fallback
```

Always validate: even schema-constrained output can contain wrong *values* (e.g. a plausible but incorrect date).

## Example
**Analogy:** Asking for structured output is like giving someone a form with labelled boxes instead of a blank page — the answers land exactly where your system expects them.

Tool-forcing pattern with the Anthropic SDK + Pydantic:

```python
# pip install anthropic pydantic
import anthropic
from pydantic import BaseModel, ValidationError

class Ticket(BaseModel):
    category: str
    priority: int        # 1 (low) – 3 (high)
    summary: str

client = anthropic.Anthropic()
tool = {
    "name": "record_ticket",
    "description": "Record a classified support ticket.",
    "input_schema": Ticket.model_json_schema(),   # JSON Schema from Pydantic
}

resp = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=500,
    tools=[tool],
    tool_choice={"type": "tool", "name": "record_ticket"},  # force this tool
    messages=[{"role": "user", "content":
               "Classify: 'App crashes every time I upload a photo, urgent!'"}],
)

tool_use = next(b for b in resp.content if b.type == "tool_use")
try:
    ticket = Ticket.model_validate(tool_use.input)   # dict -> validated object
    print(ticket)
except ValidationError as e:
    print("Invalid output, retry:", e)
# Expected (values may vary):
# category='bug' priority=3 summary='App crashes when uploading a photo'
```

The model never "replies" in prose here — it fills the tool's arguments, which the API returns as a parsed dict in `tool_use.input`.

## Common Mistakes / Misconceptions
- **Regex-scraping JSON out of prose.** Brittle; use tools or a native schema mode.
- **Trusting output without validation.** Validate types, ranges and enums; retry or fall back on failure.
- **Confusing JSON mode with schema compliance.** Valid JSON can still miss required fields.
- **Overly complex schemas.** Deeply nested, huge schemas increase errors — keep them small and descriptive (use field descriptions).

## Key Takeaways
- Structured output = LLM returns data matching a schema your code can consume.
- Most reliable: forced tool call with `input_schema` or native JSON-schema modes.
- JSON mode only guarantees syntax.
- Always validate with Pydantic (or jsonschema) and handle failures.

## Related Questions
- [What is JSON mode?](16-What-is-JSON-mode.md)
- [What is JSON?](../Week-05-Python-for-LLM-Engineering/09-What-is-JSON.md)
- [What is object-oriented programming?](../Week-05-Python-for-LLM-Engineering/06-What-is-object-oriented-programming.md)
- [What is the temperature parameter?](12-What-is-the-temperature-parameter.md)
