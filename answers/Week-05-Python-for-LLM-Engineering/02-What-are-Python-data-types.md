# What are Python data types?

> Week 05 – Python for LLM Engineering · Question 2

## In One Line
A data type tells Python **what kind of value** something is, like text, a whole number or a list.

## Simple Explanation
Every value in Python has a type. The type decides what you can do with it. You can add two numbers, but you can't add a number to a piece of text.

The most common types are:

1. **`int`**: whole numbers, like `1024`.
2. **`float`**: decimal numbers, like `0.7`.
3. **`bool`**: `True` or `False`.
4. **`str`** (string): text, like `"Hello"`.
5. **`list`**: an ordered collection, like `[1, 2, 3]`.
6. **`tuple`**: like a list, but it can't be changed.
7. **`dict`** (dictionary): pairs of name and value, like `{"role": "user"}`.
8. **`None`**: means "no value here".

Two simple rules:
- **You don't declare types.** Python works out the type from the value. `x = 5` makes an `int`.
- **Python won't mix types for you.** `"5" + 5` gives an error instead of guessing.

**Link to LLMs:** prompts and replies are `str`, chat messages are `dict`s inside a `list`, `temperature` is a `float`, and `max_tokens` is an `int`.

## Real-Life Example
Data types are like kitchen containers. Soup goes in a bowl, eggs go in a carton. If you pour soup into an egg carton, you get a mess. In Python, that mess is an error.

## Code Example
```python
import json                                   # built-in tool for reading JSON text

raw = '{"model": "claude-sonnet-5-5", "tokens": 52, "temperature": 0.7, "stop": null}'
data = json.loads(raw)                        # turn the text into Python values

print(type(data))                             # the whole thing is a dict
print(type(data["model"]))                    # text -> str
print(type(data["tokens"]))                   # whole number -> int
print(type(data["temperature"]))              # decimal number -> float
print(data["stop"] is None)                   # JSON null -> None

try:
    print("Tokens: " + data["tokens"])        # mixing text and a number
except TypeError as e:
    print("TypeError:", e)                    # Python refuses to guess

# Output:
# <class 'dict'>
# <class 'str'>
# <class 'int'>
# <class 'float'>
# True
# TypeError: can only concatenate str (not "int") to str
```

**What happened:** each value from an API-style reply got its own type. Adding text to a number failed, so we would need `str(data["tokens"])` first.

## Watch Out
- `"5"` (text) and `5` (number) are different. LLMs sometimes send numbers inside quotes, so convert them with `int()`.
- Use `x is None`, not `x == None`, to check for "no value".

## Remember
- The type decides **what you can do** with a value.
- Main types: `int`, `float`, `bool`, `str`, `list`, `tuple`, `dict`, `None`.
- LLM data is mostly **text, lists and dicts**.

## Related Questions
- [What is the difference between a list and a tuple?](03-What-is-the-difference-between-a-list-and-a-tuple.md)
- [What is a dictionary?](04-What-is-a-dictionary.md)
- [What is JSON?](09-What-is-JSON.md)
- [What is structured output?](../Week-06-Working-with-LLMs-and-Model-APIs/15-What-is-structured-output.md)
