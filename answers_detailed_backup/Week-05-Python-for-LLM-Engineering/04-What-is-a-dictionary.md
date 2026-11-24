# What is a dictionary?

> Week 05 – Python for LLM Engineering · Question 4

## Short Answer
A dictionary (`dict`) is Python's built-in **key→value mapping**. You look up a value by its key in roughly constant time (O(1)), using a hash table. Keys must be hashable (strings, numbers, tuples), values can be anything, and since Python 3.7 dicts keep insertion order. In LLM engineering, dicts are everywhere because JSON objects, API request bodies and chat messages (`{"role": "user", "content": "..."}`) all map directly to dicts.

## Detailed Explanation

**Core operations:**

| Operation | Code | Notes |
|---|---|---|
| Create | `d = {"model": "claude-sonnet-5-5"}` or `dict(model=...)` | |
| Read | `d["model"]` | Raises `KeyError` if the key is missing |
| Safe read | `d.get("temperature", 1.0)` | Returns the default if missing |
| Write / update | `d["max_tokens"] = 1024`, `d.update(other)` | |
| Delete | `del d["x"]`, `d.pop("x", None)` | |
| Iterate | `for k, v in d.items():` | |
| Membership | `"model" in d` | Checks **keys** |
| Merge (3.9+) | `defaults \| overrides` | Right side wins |
| Comprehension | `{k: v for k, v in pairs}` | |

**How it works (simplified):**
```
key "role" ──hash()──► bucket 5 ──► value "user"
```
Because lookup uses the hash, keys must be immutable/hashable. A list can't be a key.

**Useful relatives:** `collections.defaultdict` (auto-creates missing values), `collections.Counter` (counting), and `TypedDict` / Pydantic models (typed dict-like structures).

## Example
**Analogy:** a dictionary is like a coat check. You hand over a ticket (key) and get your exact coat (value) back immediately, without searching every rack.

```python
import json

# Build an API request body as a dict
defaults = {"model": "claude-haiku-5-5", "max_tokens": 512, "temperature": 1.0}
overrides = {"temperature": 0.2}
request = defaults | overrides            # merge; overrides win
request["messages"] = [{"role": "user", "content": "Summarise RAG in one line."}]
print(request["temperature"])             # 0.2

# Safely read an LLM response that might lack a field
response = json.loads('{"content": [{"type": "text", "text": "RAG = search + generate."}], "usage": {"output_tokens": 9}}')
print(response["content"][0]["text"])     # RAG = search + generate.
print(response.get("stop_reason", "unknown"))   # unknown

# Count labels returned by an LLM classifier
from collections import Counter
labels = ["positive", "negative", "positive", "neutral", "positive"]
print(Counter(labels).most_common(1))     # [('positive', 3)]
```

## Common Mistakes / Misconceptions
- **Using `d[key]` on data you don't control** (LLM or API output). Use `.get()` or validate with Pydantic to avoid `KeyError`.
- **Mutating a dict while iterating over it**, which raises `RuntimeError`. Iterate over `list(d)` instead.
- **Thinking `in` checks values.** It checks keys. Use `value in d.values()` for values.
- **Sharing one dict between requests by accident** (for example a mutable default argument). Copy it with `dict(d)` or `copy.deepcopy`.

## Key Takeaways
- Dicts map hashable keys to values with fast O(1) average lookup.
- JSON objects ⇄ Python dicts, so every LLM API payload is a dict.
- Prefer `.get()` and validation for untrusted data.
- Since Python 3.7, dicts preserve insertion order.

## Related Questions
- [What are Python data types?](02-What-are-Python-data-types.md)
- [What is JSON?](09-What-is-JSON.md)
- [How do you consume a REST API using Python?](14-How-do-you-consume-a-REST-API-using-Python.md)
