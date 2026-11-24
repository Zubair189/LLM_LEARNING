# What is a dot product?

> Week 03 – Mathematics and Neural Network Foundations · Question 5

## In One Line
The dot product **multiplies two lists number-by-number and adds the results**, giving one number that shows how similar the two lists are.

## Simple Explanation
Take two vectors (lists) of the same length:

- `a = [1, 2, 3]`
- `b = [4, 5, 6]`

Steps:
1. Multiply matching positions: 1×4 = 4, 2×5 = 10, 3×6 = 18.
2. Add them up: 4 + 10 + 18 = **32**.

That single number, 32, is the dot product.

**What does the number tell us?** If you picture the vectors as arrows:
- **Big positive** → they point in a similar direction (similar).
- **Around zero** → they point in unrelated directions.
- **Negative** → they point in opposite directions.

The dot product also grows when the arrows are longer, so it mixes "how similar" with "how big".

**Where AI uses it:**
- **Neurons:** each neuron computes a dot product of its inputs and its weights.
- **Matrix multiplication:** every number in the answer is a dot product of a row and a column.
- **Attention:** a Transformer checks "how relevant is this word to that word?" with a dot product.

**Link to LLMs:** attention scores and next-word scores are both dot products.

## Real-Life Example
Your shopping basket is `[2 apples, 3 loaves]` and the prices are `[$0.50, $2.00]`. The dot product is 2×0.50 + 3×2.00 = **$7**, your total bill. A dot product is a "weighted total".

## Code Example
```python
a = [1, 2, 3]
b = [4, 5, 6]

dot = sum(x * y for x, y in zip(a, b))   # multiply pairs, then add
print(dot)                               # 32

# A query compared with three other vectors
q = [1, 0]
print(sum(x * y for x, y in zip(q, [0.9, 0.1])))   # 0.9   similar direction
print(sum(x * y for x, y in zip(q, [0, 1])))       # 0     unrelated
print(sum(x * y for x, y in zip(q, [-1, 0])))      # -1    opposite
```

**What happened:** the first line gave the dot product 32. The second part shows how the sign tells you whether two vectors agree, are unrelated, or disagree. This is a mini version of attention.

## Watch Out
- The dot product gives **one number**, not a list. Multiplying position-by-position without adding gives a list.
- Both vectors must be the **same length**.

Note: cosine similarity (Week 7) is a dot product with the lengths removed, so it only measures direction.

## Remember
- Multiply matching numbers, then add → one number.
- Bigger = more aligned; zero = unrelated; negative = opposite.
- Neurons, matrix multiplication and attention are all built on it.

## Related Questions
- [What is a vector?](../Week-03-Mathematics-and-Neural-Network-Foundations/02-What-is-a-vector.md)
- [What is an attention score?](../Week-04-Transformer-Architecture/07-What-is-an-attention-score.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
