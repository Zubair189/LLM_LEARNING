# How do Python exceptions work?

> Week 05 – Python for LLM Engineering · Question 8

## Short Answer
An exception is an object Python **raises** when something goes wrong, such as a missing key, bad JSON or a network failure. It travels up the call stack until a matching `try/except` block **catches** it. If nothing catches it, the program stops and prints a traceback. You handle exceptions with `try / except / else / finally`, raise your own with `raise`, and define custom exception classes for errors specific to your domain. LLM apps depend on this because networks fail and model output isn't always well-formed.

## Detailed Explanation

```python
try:
    risky()                       # code that might fail
except (KeyError, ValueError) as e:
    handle(e)                     # runs only if a matching exception was raised
else:
    on_success()                  # runs only if no exception was raised
finally:
    cleanup()                     # always runs (close files, release resources)
```

**Propagation:**
```
main() → answer_question() → parse_reply() → json.loads()  ✗ JSONDecodeError
   ▲ caught here?  ◄── no ◄── no ◄── raised
```

**Hierarchy (simplified):**
```
BaseException
├── KeyboardInterrupt, SystemExit      <- don't catch these casually
└── Exception
    ├── ValueError
    │   └── json.JSONDecodeError
    ├── KeyError / IndexError (LookupError)
    ├── TypeError
    ├── OSError → ConnectionError, TimeoutError
    └── your custom exceptions
```

**Good practices:**
- Catch the **most specific** exception you can actually handle.
- Use `raise NewError(...) from e` to keep the original cause.
- Use a bare `raise` inside `except` to re-raise after logging.
- Create custom exceptions (`class LLMParseError(Exception): ...`) so callers can react precisely.
- "EAFP" (easier to ask forgiveness than permission) is idiomatic Python: try the operation and handle failure, rather than checking everything first.

## Example
**Analogy:** exceptions work like a restaurant's escalation chain. A waiter who can't fix a problem passes it to the manager, then the owner. If nobody handles it, the restaurant closes (the program crashes).

```python
import json

class LLMParseError(Exception):
    """Raised when the model's reply isn't the JSON we asked for."""

def parse_sentiment(reply: str) -> str:
    try:
        data = json.loads(reply)
        label = data["sentiment"]
    except json.JSONDecodeError as e:
        raise LLMParseError(f"Not JSON: {reply[:30]!r}") from e
    except KeyError as e:
        raise LLMParseError("Missing 'sentiment' field") from e
    else:
        return label.lower()
    finally:
        print("parse attempted")

for reply in ['{"sentiment": "Positive"}', 'Sure! Here is the JSON...', '{"mood": "ok"}']:
    try:
        print("->", parse_sentiment(reply))
    except LLMParseError as e:
        print("-> failed:", e)

# parse attempted
# -> positive
# parse attempted
# -> failed: Not JSON: 'Sure! Here is the JSON...'
# parse attempted
# -> failed: Missing 'sentiment' field
```

In a real app you might respond to `LLMParseError` by retrying the request once, or by switching to structured output (a forced tool call).

## Common Mistakes / Misconceptions
- **Bare `except:` or `except Exception: pass`**, which silently hides bugs (and a bare `except:` even catches Ctrl+C). Always log or re-raise.
- **Huge `try` blocks.** Wrap only the lines that can fail, so you know what actually broke.
- **Using exceptions for normal control flow** when a simple `if` or `.get()` is clearer.
- **Losing context** by raising a new error without `from e`.

## Key Takeaways
- Exceptions propagate up the stack until caught, otherwise the program crashes with a traceback.
- `try/except/else/finally`: handle, run on success, always clean up.
- Catch specific exceptions and chain them with `from e`.
- Treat model output and network calls as untrusted, and expect them to fail sometimes.

## Related Questions
- [How do you handle API errors?](15-How-do-you-handle-API-errors.md)
- [What is JSON?](09-What-is-JSON.md)
- [What are functions?](05-What-are-functions.md)
- [What is structured output?](../Week-06-Working-with-LLMs-and-Model-APIs/15-What-is-structured-output.md)
