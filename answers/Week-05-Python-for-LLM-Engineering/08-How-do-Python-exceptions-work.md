# How do Python exceptions work?

> Week 05 – Python for LLM Engineering · Question 8

## In One Line
An exception is Python's way of saying **"something went wrong"**, and `try/except` lets you catch it instead of crashing.

## Simple Explanation
When Python hits a problem, like a missing key or broken JSON, it **raises** an exception. If nothing deals with it, the program stops and prints an error report called a **traceback**.

You can catch problems with a `try` block:

1. **`try:`** put the code that might fail here.
2. **`except SomeError:`** what to do if that kind of error happens.
3. **`else:`** (optional) runs only if nothing went wrong.
4. **`finally:`** (optional) always runs, error or not. Good for clean-up.

Common errors you will meet:
- **`KeyError`**: a dictionary key is missing.
- **`TypeError`**: wrong type, like adding text to a number.
- **`ValueError`**: right type, bad value. `json.JSONDecodeError` is one kind of this.

You can also create an error on purpose with `raise ValueError("message")`.

**Link to LLMs:** networks fail and models sometimes return text that isn't the JSON you asked for. Exceptions let your app handle that calmly.

## Real-Life Example
Think of a restaurant. If a waiter can't fix a problem, they pass it to the manager, then to the owner. If nobody handles it, the restaurant closes for the day. In Python, the "closing" is your program crashing.

## Code Example
```python
import json

def parse_reply(reply):
    try:
        data = json.loads(reply)              # this line might fail
        return data["sentiment"]              # this one too
    except json.JSONDecodeError:              # the text was not valid JSON
        return "error: not JSON"
    except KeyError:                          # the key was missing
        return "error: no 'sentiment' field"
    finally:
        print("checked:", reply)              # always runs

print(parse_reply('{"sentiment": "positive"}'))
print(parse_reply('Sure! Here it is...'))
print(parse_reply('{"mood": "ok"}'))

# Output:
# checked: {"sentiment": "positive"}
# positive
# checked: Sure! Here it is...
# error: not JSON
# checked: {"mood": "ok"}
# error: no 'sentiment' field
```

**What happened:** three LLM-style replies went in. One worked, and the other two failed in different ways, but the program kept running.

## Watch Out
- Never write `except: pass`. It hides every error, so bugs become invisible.
- Keep the `try` block small, so you know exactly which line failed.

## Remember
- **Raise** = signal a problem. **Except** = catch it.
- Catch **specific** errors like `KeyError`, not everything.
- `finally` **always** runs.

## Related Questions
- [How do you handle API errors?](15-How-do-you-handle-API-errors.md)
- [What is JSON?](09-What-is-JSON.md)
- [What are functions?](05-What-are-functions.md)
- [What is structured output?](../Week-06-Working-with-LLMs-and-Model-APIs/15-What-is-structured-output.md)
