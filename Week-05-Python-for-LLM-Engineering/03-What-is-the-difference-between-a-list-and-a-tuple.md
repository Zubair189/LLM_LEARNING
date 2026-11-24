# What is the difference between a list and a tuple?

> Week 05 – Python for LLM Engineering · Question 3

## Short Answer
Both are ordered sequences that can hold any types, but a **list is mutable** (you can add, remove or change items) and a **tuple is immutable** (fixed once created). Use lists for collections that grow or change, such as a chat history. Use tuples for fixed records, such as `(model, max_tokens)`, for function return values, and for anything that must be hashable, like a dictionary key.

## Detailed Explanation

| Feature | `list` | `tuple` |
|---|---|---|
| Syntax | `[1, 2, 3]` | `(1, 2, 3)` or `1, 2, 3` |
| Mutable | Yes (`append`, `pop`, item assignment) | No |
| Hashable (usable as dict key / set member) | No | Yes, if all items are hashable |
| Typical meaning | Many items of the same kind | A fixed record whose positions mean something |
| Memory / speed | Slightly larger | Slightly smaller, faster to create |
| Single item | `[x]` | `(x,)`, where the comma is required |

**Semantics matter more than speed:**
- A **list** says "here is a collection of similar things", such as messages, documents or embeddings.
- A **tuple** says "here is one thing with several parts", such as `(lat, lon)` or `(text, score)`.

**Immutability is shallow:** `t = ([1], 2)` cannot be reassigned element-wise, but `t[0].append(5)` works because the inner list is mutable.

**Tuple unpacking** is common:
```python
text, score = ("Paris is the capital of France.", 0.92)
```

## Example
**Analogy:** a list is a shopping list on a whiteboard (you add and erase items). A tuple is a printed receipt (the record is fixed).

```python
# Chat history: grows each turn, so use a list
history = [{"role": "user", "content": "Hi"}]
history.append({"role": "assistant", "content": "Hello! How can I help?"})
print(len(history))  # 2

# Retrieval results: each item is a fixed (doc_id, score) record, so use tuples
results = [("doc-17", 0.91), ("doc-03", 0.85)]
for doc_id, score in results:
    print(f"{doc_id}: {score:.2f}")
# doc-17: 0.91
# doc-03: 0.85

# Tuples are hashable, so they work as cache keys
cache = {}
key = ("claude-haiku-5-5", "What is RAG?")   # (model, prompt)
cache[key] = "RAG combines retrieval with generation."
print(cache[("claude-haiku-5-5", "What is RAG?")])
# RAG combines retrieval with generation.

try:
    cache[["claude-haiku-5-5", "What is RAG?"]] = "x"   # list as key
except TypeError as e:
    print("TypeError:", e)   # TypeError: unhashable type: 'list'
```

## Common Mistakes / Misconceptions
- **Writing `("item")` for a one-element tuple.** That's just a string in parentheses. Write `("item",)`.
- **Thinking tuples are "deeply" immutable.** Mutable objects inside a tuple can still change.
- **Picking tuples only for speed.** The difference is tiny. Choose based on meaning (fixed record vs growing collection).
- **Using a list as a dict key or set member.** It raises `TypeError: unhashable type`.

## Key Takeaways
- List = mutable, for growing collections. Tuple = immutable, for fixed records.
- Tuples are hashable, so they work as cache keys and in sets.
- Tuple unpacking (`a, b = pair`) makes code readable.
- Chat histories are lists. `(text, score)` pairs from retrieval are tuples.

## Related Questions
- [What are Python data types?](02-What-are-Python-data-types.md)
- [What is a dictionary?](04-What-is-a-dictionary.md)
- [What are functions?](05-What-are-functions.md)
