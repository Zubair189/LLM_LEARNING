# What is JSON?

> Week 05 – Python for LLM Engineering · Question 9

## Short Answer
JSON (JavaScript Object Notation) is a lightweight, text-based format for structured data. It is built from objects (`{}`), arrays (`[]`), strings, numbers, booleans and `null`. It is the standard format of web APIs: every LLM API request and response is JSON, and asking a model for JSON output is the usual way to get machine-readable results. In Python, the built-in `json` module converts between JSON text and Python dicts and lists (`json.loads` / `json.dumps`).

## Detailed Explanation

**JSON ⇄ Python mapping:**

| JSON | Python |
|---|---|
| object `{"k": v}` | `dict` |
| array `[1, 2]` | `list` |
| string `"text"` (double quotes only) | `str` |
| number `42`, `3.14` | `int`, `float` |
| `true` / `false` | `True` / `False` |
| `null` | `None` |

**Core functions:**
- `json.loads(text)` → Python object (**s** = string). `json.load(file)` reads from a file.
- `json.dumps(obj, indent=2)` → JSON text. `json.dump(obj, file)` writes to a file.

**Strict rules (common failure points):**
- Keys must be **double-quoted strings**. Single quotes are invalid.
- No trailing commas, no comments, no `NaN`/`Infinity` in strict JSON.
- Not every Python type serialises: `datetime`, `set` and custom objects need conversion first.

**JSON in LLM work:**
1. **Transport:** request and response bodies of LLM APIs.
2. **Structured output:** asking the model to return data you can parse.
3. **Tool calling:** tool parameters are described with **JSON Schema**, and the model returns tool arguments as JSON.
4. **Datasets:** JSONL (one JSON object per line) is common for evals and fine-tuning data.

## Example
**Analogy:** JSON is like a standard shipping form that every country's post office understands. The packing and unpacking differ by country (programming language), but the form is universal.

````python
import json

# The model "almost" followed instructions: JSON wrapped in a markdown code fence
llm_reply = """```json
{"name": "Ada Lovelace", "born": 1815, "skills": ["math", "programming"], "email": null}
```"""

def extract_json(text: str) -> dict:
    """Strip optional ``` fences, then parse."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(cleaned)

person = extract_json(llm_reply)
print(person["name"], person["born"] + 10)   # Ada Lovelace 1825
print(person["email"] is None)               # True

# Python -> JSON (e.g. to log, or to write a JSONL eval file)
record = {"prompt": "Who was Ada?", "ok": True}
print(json.dumps(record))                    # {"prompt": "Who was Ada?", "ok": true}

try:
    json.loads("{'name': 'Ada'}")            # single quotes: invalid JSON
except json.JSONDecodeError as e:
    print("Invalid:", e.msg)                 # Invalid: Expecting property name enclosed in double quotes
````

For production, don't rely on cleaning up fences. Use structured output (a forced tool with an `input_schema`, or a provider's JSON mode) and validate with Pydantic.

## Common Mistakes / Misconceptions
- **Confusing JSON with a Python dict.** JSON is *text*. You must parse it with `json.loads`. `str(my_dict)` does not produce valid JSON (single quotes, `True`, `None`).
- **Trusting LLM JSON blindly.** It may include fences, extra prose, missing fields or wrong types. Parse, then validate.
- **Mixing up `loads`/`load` and `dumps`/`dump`.** The `s` variants work with strings, the others with files.
- **Non-ASCII output**: use `json.dumps(obj, ensure_ascii=False)` to keep readable Unicode.

## Key Takeaways
- JSON is the universal data format of web and LLM APIs.
- `json.loads` (text → Python) and `json.dumps` (Python → text).
- JSON's syntax is strict: double quotes, no trailing commas.
- Always validate model-produced JSON. Prefer native structured-output features.

## Related Questions
- [What is a dictionary?](04-What-is-a-dictionary.md)
- [How do Python exceptions work?](08-How-do-Python-exceptions-work.md)
- [What is structured output?](../Week-06-Working-with-LLMs-and-Model-APIs/15-What-is-structured-output.md)
- [What is JSON mode?](../Week-06-Working-with-LLMs-and-Model-APIs/16-What-is-JSON-mode.md)
