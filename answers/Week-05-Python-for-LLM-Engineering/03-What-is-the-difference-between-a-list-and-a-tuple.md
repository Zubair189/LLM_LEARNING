# What is the difference between a list and a tuple?

> Week 05 – Python for LLM Engineering · Question 3

## In One Line
Both hold items in order, but a **list can be changed** and a **tuple cannot**.

## Simple Explanation
A **list** is written with square brackets: `[1, 2, 3]`. You can add, remove or change items after you create it.

A **tuple** is written with round brackets: `(1, 2, 3)`. Once it is made, it is fixed. Things that can't be changed are called **immutable**.

When to use which:

1. **List:** a collection that grows or changes. *Example:* a chat history that gets a new message every turn.
2. **Tuple:** one small record with a fixed shape. *Example:* `(doc_id, score)` from a search.

Because a tuple can't change, Python can use it as a **dictionary key**. A list can't be a key.

**Unpacking** is a handy trick: `doc_id, score = ("doc-17", 0.91)` puts each part into its own variable.

**Link to LLMs:** the `messages` you send to an LLM are a list, because the conversation keeps growing.

## Real-Life Example
A list is a shopping list on a whiteboard: you add and wipe off items. A tuple is a printed receipt: the details are fixed once printed.

## Code Example
```python
history = ["Hi"]                    # a list: we can change it
history.append("Hello! How can I help?")  # add a new item
print(history)                      # both messages are there

point = ("doc-17", 0.91)            # a tuple: fixed (doc_id, score) pair
doc_id, score = point               # "unpack" it into two variables
print(doc_id, score)

try:
    point[1] = 0.5                  # try to change the tuple
except TypeError as e:
    print("TypeError:", e)          # not allowed

cache = {("claude-haiku-5-5", "What is RAG?"): "Search + generate."}  # tuple as a key
print(cache[("claude-haiku-5-5", "What is RAG?")])

# Output:
# ['Hi', 'Hello! How can I help?']
# doc-17 0.91
# TypeError: 'tuple' object does not support item assignment
# Search + generate.
```

**What happened:** the list grew, the tuple refused to change, and the tuple worked as a dictionary key.

## Watch Out
- A one-item tuple needs a comma: `("item",)`. Without the comma, `("item")` is just text.
- If a tuple holds a list, that inner list can still change. Only the tuple itself is fixed.

## Remember
- **List** `[ ]` = changeable, for growing collections.
- **Tuple** `( )` = fixed, for small records.
- Tuples can be dictionary keys; lists can't.

## Related Questions
- [What are Python data types?](02-What-are-Python-data-types.md)
- [What is a dictionary?](04-What-is-a-dictionary.md)
- [What are functions?](05-What-are-functions.md)
