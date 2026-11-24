# What is NumPy?

> Week 05 – Python for LLM Engineering · Question 10

## Short Answer
NumPy ("Numerical Python") is the core Python library for fast numerical computing. Its central object is the **`ndarray`**, an n-dimensional array of same-typed numbers stored in contiguous memory, which supports **vectorised** operations that run in optimised C instead of Python loops. Almost every AI library builds on NumPy or copies its API. In LLM engineering you'll use it most for **embeddings**: storing vectors, normalising them and computing cosine similarity for search and RAG.

## Detailed Explanation

**Why it's fast:**
```
Python list  : [ptr]→int  [ptr]→int  [ptr]→int   (scattered objects, type checks per item)
NumPy array  : [ 1 | 2 | 3 | 4 ]  float32, contiguous  → one C loop, SIMD-friendly
```

**Key concepts:**

| Concept | Example | Meaning |
|---|---|---|
| `shape` | `(1000, 1024)` | 1000 embeddings of dimension 1024 |
| `dtype` | `float32`, `int64` | Element type, which drives memory use |
| Vectorisation | `a * 2`, `a + b` | Element-wise operation, no Python loop |
| Broadcasting | `matrix / norms[:, None]` | Automatically stretches smaller shapes to match |
| Indexing / slicing | `a[0]`, `a[:, :10]`, `a[a > 0.5]` | Including boolean masks |
| Reductions | `a.sum(axis=1)`, `a.mean()` | Collapse along an axis |
| Linear algebra | `a @ b`, `np.linalg.norm` | Dot products and matrix multiplication |

**NumPy in the LLM stack:**
- Embedding vectors from an embedding model become NumPy arrays.
- Similarity search over a few thousand vectors is one matrix multiply. (Use a vector database for millions of vectors.)
- Token-ID arrays, logits and probability calculations (softmax) for analysis.
- PyTorch tensors convert to and from NumPy cheaply (`tensor.numpy()`, `torch.from_numpy`).

## Example
**Analogy:** a Python list is a shopping bag of mixed items you check one by one. A NumPy array is a tray of identical eggs that a machine can process a whole row at a time.

```python
import numpy as np

# Pretend embeddings (real ones have hundreds to thousands of dimensions)
docs = np.array([
    [1.0, 0.0, 0.0],   # "Paris is the capital of France"
    [0.7, 0.7, 0.0],   # "France has great food"
    [0.0, 0.0, 1.0],   # "Python is a programming language"
], dtype=np.float32)
query = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # "capital of France?"

def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

scores = normalize(docs) @ normalize(query)   # cosine similarity in one line
print(np.round(scores, 3))                    # [1.    0.707 0.   ]

top_k = np.argsort(scores)[::-1][:2]          # indices of the best 2 matches
print(top_k)                                  # [0 1]

# Softmax over model logits (numerically stable)
logits = np.array([2.0, 1.0, 0.1])
probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
print(np.round(probs, 3))                     # [0.659 0.242 0.099]
```

## Common Mistakes / Misconceptions
- **Looping in Python over arrays** (`for i in range(len(a))`). This throws away NumPy's speed. Vectorise.
- **Shape mismatches.** Print `.shape` constantly. `(n,)` and `(n, 1)` behave differently under broadcasting.
- **Unintended `float64`.** It doubles memory for large embedding sets. Use `float32` when that precision is enough.
- **Views vs copies.** Slicing returns a *view*, so modifying it changes the original. Use `.copy()` when needed.

## Key Takeaways
- NumPy's `ndarray` gives fast, vectorised maths over typed, contiguous memory.
- Learn shape, dtype, broadcasting and axis-wise reductions.
- Cosine similarity over embeddings is a one-liner: normalise, then matrix-multiply.
- It is the foundation for Pandas, scikit-learn and PyTorch interoperability.

## Related Questions
- [What is Pandas?](11-What-is-Pandas.md)
- [Why is Python widely used in AI development?](01-Why-is-Python-widely-used-in-AI-development.md)
- [What are top-k and top-p sampling?](../Week-06-Working-with-LLMs-and-Model-APIs/13-What-are-top-k-and-top-p-sampling.md)
