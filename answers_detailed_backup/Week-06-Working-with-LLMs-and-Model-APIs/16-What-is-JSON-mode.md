# What is JSON mode?

> Week 06 – Working with LLMs and Model APIs · Question 16

> Note: JSON mode is a **specific provider setting** that guarantees valid JSON syntax. [Structured output](15-What-is-structured-output.md) is the **broader goal** of getting data that matches your exact schema — JSON mode is one (weaker) way to approach it.

## Short Answer
JSON mode is a provider option that constrains the model's decoding so its response is always syntactically valid JSON — no prose before or after, no trailing commas. Examples are OpenAI's `response_format={"type": "json_object"}` and Ollama's `format="json"`. It guarantees parseable JSON but **not** a particular schema, so you still describe the fields in the prompt and validate the result, or use schema-constrained structured output when you need guaranteed fields.

## Detailed Explanation
How it works: at each generation step the inference engine masks out tokens that would make the output invalid JSON (constrained decoding), so `json.loads()` will succeed.

```
 Without JSON mode:  "Sure! Here's the data:\n```json\n{\"name\": \"Ana\"}\n```"   ← json.loads fails
 With JSON mode:     {"name": "Ana"}                                               ← json.loads works
```

| Feature | Plain prompt | JSON mode | JSON-schema / tool-based structured output |
|---|---|---|---|
| Valid JSON syntax | Not guaranteed | Guaranteed* | Guaranteed* |
| Required keys/types | Not guaranteed | Not guaranteed | Enforced by schema |
| Setup | Prompt only | One flag + mention JSON in prompt | Provide a JSON Schema |

\*Barring truncation: if `max_tokens` is hit mid-object, the JSON is incomplete — always check the stop reason.

Provider landscape (late 2026):
- **OpenAI**: `response_format={"type":"json_object"}` (JSON mode) or `{"type":"json_schema", ...}` (strict schema).
- **Ollama**: `format="json"` or `format=<JSON schema dict>`.
- **Anthropic**: no separate "JSON mode" flag; use a forced tool call with `input_schema` or its structured-outputs feature for schema-guaranteed JSON.

Rule of thumb: prefer schema-based structured output; use JSON mode when the shape is simple or the provider lacks schema support.

## Example
**Analogy:** JSON mode is like a form that only checks your handwriting is legible — it doesn't check you filled in the right boxes.

Ollama JSON mode + Pydantic validation:
```python
# Requires: ollama pull llama3.2 ; pip install requests pydantic
import json, requests
from pydantic import BaseModel, ValidationError

class Person(BaseModel):
    name: str
    age: int

r = requests.post("http://localhost:11434/api/chat", json={
    "model": "llama3.2",
    "format": "json",                     # JSON mode
    "stream": False,
    "options": {"temperature": 0},
    "messages": [{"role": "user", "content":
        'Extract as JSON with keys "name" (string) and "age" (integer): '
        "'Ana is a 31-year-old engineer.'"}],
}, timeout=120)

raw = r.json()["message"]["content"]
data = json.loads(raw)                    # guaranteed to parse
print(data)                               # Expected: {'name': 'Ana', 'age': 31}
try:
    print(Person.model_validate(data))    # Expected: name='Ana' age=31
except ValidationError as e:
    print("Valid JSON but wrong shape:", e)
```

Swap to schema enforcement in Ollama by passing `"format": Person.model_json_schema()` instead of `"json"`.

## Common Mistakes / Misconceptions
- **"JSON mode means my schema is followed."** It only means the output parses.
- **Not mentioning JSON in the prompt.** Some providers require it, and without guidance the model may emit an empty or odd object.
- **Ignoring truncation.** A response cut by `max_tokens` is invalid JSON.
- **Wrapping in markdown fences then parsing.** With JSON mode you get raw JSON — no need to strip fences; without it, you might.

## Key Takeaways
- JSON mode = constrained decoding for valid JSON syntax.
- It does not enforce keys or types — validate with Pydantic.
- Schema-based structured output (tools or JSON-schema modes) is stronger.
- Check `stop_reason`/`done_reason` for truncation.

## Related Questions
- [What is structured output?](15-What-is-structured-output.md)
- [What is JSON?](../Week-05-Python-for-LLM-Engineering/09-What-is-JSON.md)
- [What is Ollama?](05-What-is-Ollama.md)
- [How do Python exceptions work?](../Week-05-Python-for-LLM-Engineering/08-How-do-Python-exceptions-work.md)
