# What are functions?

> Week 05 – Python for LLM Engineering · Question 5

## Short Answer
A function is a named, reusable block of code that takes inputs (**parameters**), does some work, and optionally **returns** a value. Functions let you avoid repetition, give names to ideas such as `build_prompt()` or `call_llm()`, and test pieces in isolation. In Python, functions are first-class objects: you can pass them as arguments, return them, and store them. Retry decorators and LLM tool-calling both rely on this.

## Detailed Explanation

```python
def name(positional, default=1, *args, keyword_only=None, **kwargs) -> ReturnType:
    """Docstring explaining what it does."""
    ...
    return value
```

| Concept | Meaning |
|---|---|
| **Parameters vs arguments** | Parameters are the names in the definition. Arguments are the values passed in a call. |
| **Default values** | `def f(temperature=1.0)` makes the argument optional. |
| **Keyword arguments** | `f(temperature=0.2)` is clearer than positional arguments. |
| `*args` / `**kwargs` | Collect extra positional or keyword arguments. Useful for passing options through to an SDK. |
| **Return** | Without `return`, a function returns `None`. |
| **Scope** | Variables created inside a function are local to it. |
| **Type hints + docstring** | Document the contract. Tool-calling frameworks often turn these into JSON schemas. |
| **Lambda** | Small anonymous function: `key=lambda r: r["score"]`. |
| **Decorator** | A function that wraps another to add behaviour (logging, retries, caching). |

**Pure functions** (same input gives the same output, no side effects) are easy to test. Keep prompt building pure, and isolate network calls in separate functions.

## Example
**Analogy:** a function is like a recipe card. Give it ingredients (arguments), follow the steps, get a dish (return value). You can reuse the card any number of times.

```python
import time
from functools import wraps

def build_prompt(question: str, context: list[str], max_chunks: int = 2) -> str:
    """Pure function: combine retrieved chunks and a question into a prompt."""
    joined = "\n".join(f"- {c}" for c in context[:max_chunks])
    return f"Answer using only this context:\n{joined}\n\nQuestion: {question}"

def timed(func):
    """Decorator: report how long a function call took."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start:.3f}s")
        return result
    return wrapper

@timed
def fake_llm(prompt: str) -> str:   # stand-in for a real API call
    return f"[answer based on {prompt.count('- ')} chunks]"

prompt = build_prompt("What is the capital of France?",
                      ["Paris is the capital of France.", "France is in Europe.", "Unrelated."])
print(prompt)
print(fake_llm(prompt))
# Answer using only this context:
# - Paris is the capital of France.
# - France is in Europe.
#
# Question: What is the capital of France?
# fake_llm took 0.000s
# [answer based on 2 chunks]
```

## Common Mistakes / Misconceptions
- **Mutable default arguments.** `def add(msg, history=[])` shares *one* list across all calls. Use `history=None` and create a new list inside.
- **Forgetting `return`**, so the function silently returns `None`.
- **Giant "do everything" functions** that build the prompt, call the API, parse the result and save it, all in one. Split them so each piece is testable.
- **Confusing `print` with `return`.** Printing shows a value. Returning lets other code use it.

## Key Takeaways
- Functions package reusable logic behind a clear name and signature.
- Type hints and docstrings document the contract, and LLM tool-calling schemas are often generated from them.
- Functions are first-class, which makes decorators like retry/timing wrappers possible.
- Keep prompt construction pure and isolate side effects such as API calls.

## Related Questions
- [What is object-oriented programming?](06-What-is-object-oriented-programming.md)
- [How do Python exceptions work?](08-How-do-Python-exceptions-work.md)
- [How do you handle API errors?](15-How-do-you-handle-API-errors.md)
