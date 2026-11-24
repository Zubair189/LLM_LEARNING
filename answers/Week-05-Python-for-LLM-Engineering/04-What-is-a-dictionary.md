# What is a dictionary?

> Week 05 – Python for LLM Engineering · Question 4

## In One Line
A dictionary stores **pairs of a key and a value**, so you can find any value quickly by its key.

## Simple Explanation
A dictionary (`dict`) is written with curly brackets: `{"role": "user", "content": "Hi"}`. Here `"role"` is a **key** (the label) and `"user"` is its **value** (the data).

How you use it:

1. **Read:** `d["role"]` gives `"user"`.
2. **Add or change:** `d["temperature"] = 0.2`.
3. **Read safely:** `d.get("stop", "not set")` returns a default instead of crashing when the key is missing.
4. **Check a key:** `"role" in d` gives `True` or `False`.
5. **Loop:** `for key, value in d.items():` goes through every pair.

Looking up a key is very fast, even in a huge dictionary, because Python jumps straight to it instead of searching one by one. Keys must be things that can't change, like text, numbers or tuples.

**Link to LLMs:** JSON objects turn into Python dictionaries. Every API request, API reply and chat message (`{"role": ..., "content": ...}`) is a dictionary.

## Real-Life Example
A dictionary is like a coat check. You hand over your ticket number (the key) and get your exact coat (the value) back straight away. Nobody searches every rack.

## Code Example
```python
request = {"model": "claude-haiku-5-5", "max_tokens": 512}  # create a dict

print(request["model"])                     # read a value by its key
request["temperature"] = 0.2                # add a new key-value pair
print(request)

print(request.get("stop", "not set"))       # safe read: default if missing
print("model" in request)                   # does the key exist?

message = {"role": "user", "content": "Hi"} # one chat message is a dict
for key, value in message.items():          # loop over every pair
    print(key, "->", value)

# Output:
# claude-haiku-5-5
# {'model': 'claude-haiku-5-5', 'max_tokens': 512, 'temperature': 0.2}
# not set
# True
# role -> user
# content -> Hi
```

**What happened:** we built an API request as a dict, added a setting, read a missing key safely with `.get()`, and looped over a chat message.

## Watch Out
- `d["missing"]` crashes with `KeyError`. For data from an API or LLM, prefer `.get()`.
- `in` checks **keys**, not values.

## Remember
- A dict = **key → value** pairs.
- Lookups by key are **fast**.
- JSON objects and chat messages are **dicts** in Python.

## Related Questions
- [What are Python data types?](02-What-are-Python-data-types.md)
- [What is JSON?](09-What-is-JSON.md)
- [How do you consume a REST API using Python?](14-How-do-you-consume-a-REST-API-using-Python.md)
