# What is structured output?

> Week 06 – Working with LLMs and Model APIs · Question 15

> Note: **Structured output** means getting data in the exact shape you need, while **JSON mode** ([Q16](16-What-is-JSON-mode.md)) only promises the text is valid JSON, not that it has the right fields.

## In One Line
Structured output means making the LLM **reply in a fixed, machine-readable shape** (usually JSON with set fields) so your code can use it directly.

## Simple Explanation
Normal LLM answers are free text, like "The sentiment is mostly positive, I'd say...". That's fine for people, but hard for code.

With structured output you ask for something like `{"category": "bug", "priority": 3}`. Now your code can read `priority` straight away.

To describe the shape, you use a **schema**: a description of which fields must exist and what type each one is (text, number and so on).

Ways to get it, from weakest to strongest:

1. **Just ask in the prompt:** "Reply in JSON with keys..." Often works, but not guaranteed.
2. **JSON mode:** a provider setting that guarantees valid JSON, but not the right fields.
3. **Schema-based:** give the model the schema and make it follow it. With Anthropic, you define a **tool** with an `input_schema` and force the model to use it, or use its structured-outputs feature.

**Always check the result.** A library called **Pydantic** compares the data to your schema and raises an error if something is wrong. Then you can retry.

**Link to LLMs:** structured output turns an LLM into something like a function: text in, neat data out.

## Real-Life Example
Asking for structured output is like handing someone a form with labelled boxes instead of a blank page. The answers land exactly where your system expects them.

## Code Example
Checking LLM output against a schema with Pydantic.

```python
from pydantic import BaseModel, ValidationError     # pip install pydantic

class Ticket(BaseModel):                            # the shape we want
    category: str
    priority: int                                   # must be a whole number

good = {"category": "bug", "priority": 3}           # pretend LLM output #1
bad = {"category": "bug", "priority": "very high"}  # pretend LLM output #2

print(Ticket.model_validate(good))                  # passes: becomes a Ticket object

try:
    Ticket.model_validate(bad)                      # wrong type for priority
except ValidationError as e:
    print("Rejected:", e.errors()[0]["msg"])

# Output:
# category='bug' priority=3
# Rejected: Input should be a valid integer, unable to parse string as an integer
```

**What happened:** the good reply became a typed `Ticket` object. The bad one was caught before it could break the rest of the app. `Ticket.model_json_schema()` can turn this same class into the schema you give the model.

## Watch Out
- Valid JSON is not the same as **correct** data. It can still miss fields or have wrong values.
- Don't pull JSON out of chatty text with tricks. Use tools or schema features instead.

## Remember
- Structured output = **data in a fixed shape**.
- Strongest way: **schema + forced tool / structured-outputs feature**.
- **Always validate** (for example with Pydantic).

## Related Questions
- [What is JSON mode?](16-What-is-JSON-mode.md)
- [What is JSON?](../Week-05-Python-for-LLM-Engineering/09-What-is-JSON.md)
- [What is object-oriented programming?](../Week-05-Python-for-LLM-Engineering/06-What-is-object-oriented-programming.md)
- [What is the temperature parameter?](12-What-is-the-temperature-parameter.md)
