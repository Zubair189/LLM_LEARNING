# What is a matrix?

> Week 03 – Mathematics and Neural Network Foundations · Question 3

## Short Answer
A matrix is a rectangular grid of numbers arranged in rows and columns, with shape `(m, n)`. You can treat it as data (a table, or a batch of vectors stacked as rows) or as a function that transforms vectors: multiplying a vector by a matrix maps it to a new vector. Neural-network weights are stored as matrices, and almost all of an LLM's computation is matrix multiplication.

## Detailed Explanation
$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \text{shape } (2, 3)\text{: 2 rows, 3 columns}$$

Entry $A_{ij}$ is the value in row $i$, column $j$ (e.g. $A_{12} = 2$).

**Key operations**
- **Transpose** $A^T$: swap rows and columns, so shape `(2,3)` becomes `(3,2)`.
- **Matrix–vector product** $A\mathbf{x}$: each output element is the dot product of one row of $A$ with $\mathbf{x}$. A `(2,3)` matrix maps 3-D vectors to 2-D vectors.
- **Matrix–matrix product** $AB$: requires `A.shape[1] == B.shape[0]`. Output shape is `(A rows, B cols)`. Entry $(AB)_{ij}$ = row $i$ of $A$ · column $j$ of $B$.
- **Element-wise (Hadamard) product** $A \odot B$: same shapes, multiply entry-by-entry. This is *not* the same as matrix multiplication.

**Shape rule to memorise:** `(m, k) @ (k, n) -> (m, n)`. The inner dimensions must match and then disappear.

**Matrices in LLMs**
| Matrix | Typical shape | Role |
|---|---|---|
| Embedding table | `(vocab_size, d_model)` | one row per token |
| $W_Q, W_K, W_V$ | `(d_model, d_head)` | project hidden states to query, key and value vectors |
| Attention scores | `(seq_len, seq_len)` | how much each token attends to each other token |
| MLP weights | `(d_model, 4·d_model)` | feed-forward transformation |

## Example
**Analogy:** A spreadsheet of 3 students × 4 exam scores is a `(3, 4)` matrix. A "grading policy" matrix that turns 4 scores into 2 final grades (theory, practice) is a `(4, 2)` matrix. Multiplying them gives a `(3, 2)` table of final grades for every student at once.

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.shape, A.T.shape)    # (2, 3) (3, 2)

x = np.array([1, 0, -1])
print(A @ x)                 # [-2 -2]   row1: 1-3, row2: 4-6

B = np.array([[1, 0],
              [0, 1],
              [1, 1]])
print(A @ B)                 # [[ 4  5]
                             #  [10 11]]   (2,3) @ (3,2) -> (2,2)
```
Hand check for entry (0,0): row `[1,2,3]` · column `[1,0,1]` = 1 + 0 + 3 = 4.

## Common Mistakes / Misconceptions
- **Using `*` when you mean `@`.** In NumPy, `A * B` is element-wise and `A @ B` is matrix multiplication.
- **Assuming $AB = BA$.** Matrix multiplication is not commutative, and often only one order is even shape-compatible.
- **Ignoring shapes.** Most bugs in model code are shape mismatches. Write the shape next to every line while learning.
- **Row-vs-column conventions.** Papers often write $W\mathbf{x}$ (column vectors), while code usually writes `x @ W` (row vectors). They are the same idea with transposed weights.

## Key Takeaways
- A matrix is a 2-D grid of numbers with shape `(rows, cols)`.
- It is both a data container (a batch of vectors) and a linear transformation.
- `(m,k) @ (k,n) -> (m,n)`. Always check that the inner dimensions match.
- Model weights are matrices, and running a model is mostly multiplying by them.

## Related Questions
- [What is a vector?](../Week-03-Mathematics-and-Neural-Network-Foundations/02-What-is-a-vector.md)
- [Why do neural networks use matrices?](../Week-03-Mathematics-and-Neural-Network-Foundations/04-Why-do-neural-networks-use-matrices.md)
- [What is a dot product?](../Week-03-Mathematics-and-Neural-Network-Foundations/05-What-is-a-dot-product.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
