# What is a matrix?

> Week 03 – Mathematics and Neural Network Foundations · Question 3

## In One Line
A matrix is a **grid of numbers** arranged in rows and columns, like a small spreadsheet.

## Simple Explanation
If a vector is one list of numbers, a matrix is **several lists stacked on top of each other**.

```
A = [[1, 2, 3],
     [4, 5, 6]]
```

This matrix has 2 rows and 3 columns, so we say its **shape** is `(2, 3)`.

A matrix can be used in two ways:

1. **To hold data:** each row is one item. For example, 3 students × 4 test scores is a `(3, 4)` matrix.
2. **To transform data:** multiplying a vector by a matrix turns it into a new vector. This is what a neural-network layer does.

**How matrix multiplication works (step by step):**
- Take one **row** from the first matrix and one **column** from the second.
- Multiply matching numbers and add them up (this is a dot product).
- That single number goes into the answer.

**The shape rule:** `(2, 3) × (3, 2) → (2, 2)`. The two inner numbers (here 3 and 3) **must match**, and they disappear in the answer.

**Link to LLMs:** a model's **weights** (the numbers it learns) are stored in matrices. Running an LLM is mostly multiplying by these matrices, again and again.

## Real-Life Example
A shop has a table of orders (each row a customer, each column how many burgers and fries they bought). Multiply it by a small price table and you get every customer's bill at once. One matrix multiplication does all the sums for you.

## Code Example
```python
A = [[1, 2, 3],
     [4, 5, 6]]          # shape (2, 3)
B = [[1, 0],
     [0, 1],
     [1, 1]]             # shape (3, 2)

# Build the answer: row of A times column of B, summed
C = [[sum(A[i][k] * B[k][j] for k in range(3))   # dot product
      for j in range(2)]                         # each column of B
     for i in range(2)]                          # each row of A
print(C)                 # [[4, 5], [10, 11]]
```

Hand check for the top-left number: row `[1, 2, 3]` · column `[1, 0, 1]` = 1 + 0 + 3 = **4**.

**What happened:** a `(2, 3)` matrix times a `(3, 2)` matrix gave a `(2, 2)` answer. Each answer number is "row times column, then add up". In NumPy you would simply write `A @ B`.

## Watch Out
- **Order matters:** `A × B` is usually not the same as `B × A`, and often only one order even works.
- In NumPy, `A * B` multiplies number-by-number. `A @ B` is real matrix multiplication.

## Remember
- Matrix = numbers in **rows and columns**; shape = `(rows, columns)`.
- Multiplication = "row times column, then add".
- Model weights are matrices, so LLMs run on matrix multiplication.

## Related Questions
- [What is a vector?](../Week-03-Mathematics-and-Neural-Network-Foundations/02-What-is-a-vector.md)
- [Why do neural networks use matrices?](../Week-03-Mathematics-and-Neural-Network-Foundations/04-Why-do-neural-networks-use-matrices.md)
- [What is a dot product?](../Week-03-Mathematics-and-Neural-Network-Foundations/05-What-is-a-dot-product.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
