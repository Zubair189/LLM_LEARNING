# What are Python data types?

> Week 05 – Python for LLM Engineering · Question 2

## Short Answer
A data type defines what kind of value an object holds and which operations it supports. Python's core built-in types are numbers (`int`, `float`, `bool`), text (`str`), sequences (`list`, `tuple`), mappings (`dict`), sets (`set`), and `None`. Python is **dynamically typed** (types belong to values, not variables), but you can add **type hints** that tools like mypy and Pydantic use to catch mistakes.

## Detailed Explanation

| Category | Type | Example | Mutable? |
|---|---|---|---|
| Numeric | `int`, `float`, `complex` | `1024`, `0.7` | No |
| Boolean | `bool` | `True` | No |
| Text | `str` | `"Hello"` | No |
| Binary | `bytes`, `bytearray` | `b"\x00"` | `bytes` no, `bytearray` yes |
| Sequence | `list`, `tuple`, `range` | `[1, 2]`, `(1, 2)` | list yes, tuple no |
| Mapping | `dict` | `{"role": "user"}` | Yes |
| Set | `set`, `frozenset` | `{"a", "b"}` | set yes |
| Null | `NoneType` | `None` | — |

**Key ideas:**
- **Dynamic typing:** `x = 5` then `x = "five"` is allowed. Variables are labels pointing at objects.
- **Strong typing:** `"5" + 5` raises `TypeError`. Python won't silently convert.
- **Mutability:** mutable objects (list, dict, set) can change in place. Immutable ones (str, tuple, int) cannot. This matters for default arguments and dictionary keys.
- **Type hints** (`def f(x: str) -> int:`) are not enforced at runtime by default, but editors, `mypy` and Pydantic use them.

**Where they appear in LLM work:**
- `str`: prompts and completions
- `dict` / `list`: JSON request and response bodies (`messages=[{"role": ..., "content": ...}]`)
- `float`: `temperature`, scores and embeddings
- `int`: `max_tokens` and token counts
- `None`: optional fields such as a missing `stop_reason`

## Example
**Analogy:** data types are like containers in a kitchen. You pour soup into a bowl (a `str` holds text) and keep eggs in a carton (a `list` holds ordered items). Using the wrong container causes a mess (a `TypeError`).

```python
import json

# A (simplified) raw LLM API response body as text
raw = '{"model": "claude-sonnet-5-5", "usage": {"input_tokens": 12, "output_tokens": 40}, "stop_reason": null}'
data = json.loads(raw)

print(type(data))                            # <class 'dict'>
print(type(data["model"]))                   # <class 'str'>
print(type(data["usage"]["input_tokens"]))   # <class 'int'>
print(data["stop_reason"] is None)           # True

total: int = data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
print(f"Total tokens: {total}")              # Total tokens: 52

# Strong typing in action:
try:
    "Tokens: " + total
except TypeError as e:
    print("TypeError:", e)   # TypeError: can only concatenate str (not "int") to str
```

## Common Mistakes / Misconceptions
- **Assuming type hints are enforced.** `def f(x: int)` still accepts `"hi"` at runtime unless you validate, for example with Pydantic.
- **Comparing floats with `==`.** `0.1 + 0.2 == 0.3` is `False`. Use `math.isclose`.
- **Using `x == None` instead of `x is None`.**
- **Forgetting that JSON numbers can come back as `str`** if the LLM put quotes around them. Validate and convert.

## Key Takeaways
- Python is dynamically **and** strongly typed.
- Know which types are mutable (list, dict, set) and which are immutable (str, tuple, int).
- LLM API payloads are mostly nested `dict`/`list`/`str`/`int`/`float` structures.
- Use type hints plus Pydantic to make types explicit and validated.

## Related Questions
- [What is the difference between a list and a tuple?](03-What-is-the-difference-between-a-list-and-a-tuple.md)
- [What is a dictionary?](04-What-is-a-dictionary.md)
- [What is JSON?](09-What-is-JSON.md)
- [What is structured output?](../Week-06-Working-with-LLMs-and-Model-APIs/15-What-is-structured-output.md)
