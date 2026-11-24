# What is NumPy?

> Week 05 – Python for LLM Engineering · Question 10

## In One Line
NumPy is a Python library for doing **fast maths on whole lists of numbers at once**.

## Simple Explanation
A normal Python list can hold numbers, but doing maths on it means looping over items one by one, which is slow.

NumPy gives you a special list called an **array**. All items in an array are the same type of number, and they sit side by side in memory. This lets NumPy do the maths in fast C code.

Key ideas:

1. **Vectorised maths:** `prices * 2` doubles every number at once. No loop needed.
2. **Shape:** the size of an array. `(3, 2)` means 3 rows with 2 numbers each.
3. **Dot product (`@`):** multiplies two arrays and adds the results. It is used to measure how similar two lists of numbers are.

Most AI libraries, like Pandas and PyTorch, are built on NumPy or copy its style.

**Link to LLMs:** an **embedding** is a list of numbers that represents the meaning of a text. NumPy is often used to compare embeddings and find the most similar document, which is the heart of search and RAG.

## Real-Life Example
A Python list is like a bag of mixed shopping you check item by item. A NumPy array is like a tray of identical eggs that a machine can stamp a whole row at a time.

## Code Example
```python
import numpy as np                              # the usual short name

prices = np.array([10, 20, 30])                 # make an array
print(prices * 2)                               # maths on every item at once

docs = np.array([[1.0, 0.0],                    # pretend embedding of doc 0
                 [0.7, 0.7],                    # doc 1
                 [0.0, 1.0]])                   # doc 2
query = np.array([1.0, 0.0])                    # pretend embedding of the question

scores = docs @ query                           # dot product: how similar is each doc?
print(scores)                                   # bigger = more similar
print(np.argmax(scores))                        # index of the best match
print(docs.shape)                               # 3 rows, 2 numbers each

# Output:
# [20 40 60]
# [1.  0.7 0. ]
# 0
# (3, 2)
```

**What happened:** NumPy doubled all prices in one step, then scored three tiny "documents" against a question and picked doc 0 as the best match. Real embeddings have hundreds or thousands of numbers, but the idea is the same.

## Watch Out
- Don't loop over a NumPy array with `for`. That throws away its speed. Use array maths.
- Check `.shape` often. Most NumPy errors come from shapes that don't match.

## Remember
- NumPy = **fast maths on arrays**.
- Do maths on the **whole array at once**.
- Used for **embeddings and similarity** in LLM apps.

## Related Questions
- [What is Pandas?](11-What-is-Pandas.md)
- [Why is Python widely used in AI development?](01-Why-is-Python-widely-used-in-AI-development.md)
- [What are top-k and top-p sampling?](../Week-06-Working-with-LLMs-and-Model-APIs/13-What-are-top-k-and-top-p-sampling.md)
