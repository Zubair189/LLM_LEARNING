# What is a vector?

> Week 03 – Mathematics and Neural Network Foundations · Question 2

## In One Line
A vector is an **ordered list of numbers**, like `[3, 4]`, that describes one thing.

## Simple Explanation
A vector is a list of numbers where **each position has a fixed meaning**.

For example, a house could be described as `[bedrooms, size, age] = [3, 120, 15]`. Position 1 is always bedrooms, position 2 is always size, and so on. If you swap the order, you describe a different house.

There are two easy ways to picture a vector:

1. **As a description:** a list of features about one object (like the house above).
2. **As an arrow:** `[3, 4]` is an arrow from the starting point (0, 0) to the point 3 steps right and 4 steps up. Its **length** is 5 (from Pythagoras: √(3² + 4²) = 5).

The number of items in the list is called its **dimension**. `[3, 4]` has dimension 2.

**Things we often do with vectors:**
- **Add** two vectors: add matching positions. `[1, 2] + [3, 4] = [4, 6]`.
- **Scale** a vector: multiply every number by one value.
- **Compare** two vectors: check if they point in a similar direction (the dot product, Question 5).

**Link to LLMs:** every word piece (token) is turned into a vector of thousands of numbers, called an **embedding**. Words with similar meanings get vectors that point in similar directions.

## Real-Life Example
Describing a movie by how much action, romance and comedy it has: `[0.9, 0.1, 0.4]`. Another movie with `[0.8, 0.2, 0.5]` is a similar movie, because the lists are close. That is exactly how AI finds "similar" things.

## Code Example
```python
import math

v = [3, 4]                          # a 2-number vector
w = [1, 2]                          # another vector

length = math.sqrt(3**2 + 4**2)     # length of v
print(length)                       # 5.0

added = [a + b for a, b in zip(v, w)]   # add matching positions
print(added)                        # [4, 6]

scaled = [2 * a for a in v]         # multiply every number by 2
print(scaled)                       # [6, 8]
```

**What happened:** we measured the vector's length, added two vectors position by position, and scaled one by 2.

## Watch Out
- **Dimension is not length.** `[3, 4]` has dimension 2 (two numbers) but length 5.
- **Order matters.** `[3, 120]` and `[120, 3]` are different vectors.

Note: Week 7 Q3 covers how *text* is turned into vectors; this question covers the basic idea of a vector itself.

## Remember
- Vector = an **ordered list** of numbers.
- It can be seen as a description or as an arrow with a length and direction.
- LLMs store every token as a vector (embedding).

## Related Questions
- [What is a scalar?](../Week-03-Mathematics-and-Neural-Network-Foundations/01-What-is-a-scalar.md)
- [What is a dot product?](../Week-03-Mathematics-and-Neural-Network-Foundations/05-What-is-a-dot-product.md)
- [What are token embeddings?](../Week-04-Transformer-Architecture/10-What-are-token-embeddings.md)
- [What is a vector representation?](../Week-07-Embeddings-and-Vector-Databases/03-What-is-a-vector-representation.md)
