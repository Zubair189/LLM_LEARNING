# What is JSON?

> Week 05 – Python for LLM Engineering · Question 9

## In One Line
JSON is a simple **text format for sending structured data** between programs, and every LLM API uses it.

## Simple Explanation
JSON (JavaScript Object Notation) looks a lot like a Python dictionary, but it is just **text**. Because it is plain text, any program in any language can read it.

JSON has only a few building blocks:

1. **Objects** `{"key": value}` become Python **dicts**.
2. **Arrays** `[1, 2]` become Python **lists**.
3. **Strings** `"text"` become **str**.
4. **Numbers** `42`, `3.14` become **int** / **float**.
5. **`true` / `false`** become **`True` / `False`**.
6. **`null`** becomes **`None`**.

Python's built-in `json` module does the conversion:
- **`json.loads(text)`**: JSON text → Python (think "load string").
- **`json.dumps(obj)`**: Python → JSON text (think "dump string").

JSON is strict: keys need **double quotes**, and extra commas at the end are not allowed.

**Link to LLMs:** every API request and reply is JSON. You also often ask the model to answer in JSON so your code can read the result.

## Real-Life Example
JSON is like a standard postal form that every country understands. Each country (programming language) packs and unpacks it in its own way, but the form itself is the same everywhere.

## Code Example
```python
import json                                  # built-in JSON module

reply = '{"name": "Ada", "born": 1815, "skills": ["math"], "email": null}'  # JSON text
person = json.loads(reply)                   # text -> Python dict
print(person["name"], person["born"] + 10)   # now it's real data we can use
print(person["email"] is None)               # JSON null became None

record = {"prompt": "Who was Ada?", "ok": True}
print(json.dumps(record))                    # Python dict -> JSON text

try:
    json.loads("{'name': 'Ada'}")            # single quotes are not allowed
except json.JSONDecodeError as e:
    print("Invalid:", e.msg)

# Output:
# Ada 1825
# True
# {"prompt": "Who was Ada?", "ok": true}
# Invalid: Expecting property name enclosed in double quotes
```

**What happened:** we turned JSON text into a dict, turned a dict back into JSON (note `True` became `true`), and saw that single quotes are invalid JSON.

## Watch Out
- Don't trust LLM JSON blindly. Models sometimes add extra words or wrap the JSON in a markdown code fence, so always parse it inside `try/except`.
- `str(my_dict)` is **not** JSON (it uses single quotes and `True`). Use `json.dumps`.

## Remember
- JSON = **structured data as text**.
- `json.loads` reads it, `json.dumps` writes it.
- LLM APIs **send and receive JSON**.

## Related Questions
- [What is a dictionary?](04-What-is-a-dictionary.md)
- [How do Python exceptions work?](08-How-do-Python-exceptions-work.md)
- [What is structured output?](../Week-06-Working-with-LLMs-and-Model-APIs/15-What-is-structured-output.md)
- [What is JSON mode?](../Week-06-Working-with-LLMs-and-Model-APIs/16-What-is-JSON-mode.md)
