# What is a dot product?

> Week 03 – Mathematics and Neural Network Foundations · Question 5

## Short Answer
The dot product of two equal-length vectors is the sum of their element-wise products: $\mathbf{a}\cdot\mathbf{b} = \sum_i a_i b_i$. The result is a single scalar that measures how much the vectors point in the same direction, scaled by their lengths. It is the basic operation of neural networks: each neuron computes a dot product, and Transformer attention scores are dot products between queries and keys.

## Detailed Explanation
**Algebraic definition:**
$$\mathbf{a}\cdot\mathbf{b} = a_1b_1 + a_2b_2 + \dots + a_nb_n$$

**Geometric definition:**
$$\mathbf{a}\cdot\mathbf{b} = \|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta$$
where $\theta$ is the angle between the vectors. So the sign and size tell you about alignment:

| Angle | cos θ | Dot product | Interpretation |
|---|---|---|---|
| 0° | 1 | large positive | same direction (similar) |
| 90° | 0 | 0 | orthogonal (unrelated) |
| 180° | −1 | large negative | opposite |

**Where it appears in LLM engineering**
- **Neurons:** $z = \mathbf{w}\cdot\mathbf{x} + b$. The neuron "fires" when the input aligns with its weight pattern.
- **Matrix multiplication:** every entry of $AB$ is a dot product of a row and a column.
- **Attention:** score$(q, k) = \mathbf{q}\cdot\mathbf{k}/\sqrt{d_k}$, i.e. "how relevant is this key to my query?"
- **Output logits:** the final hidden state is dotted with each token's output embedding to score every possible next token.
- **Similarity search:** cosine similarity is the dot product of normalised vectors: $\cos\theta = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$.

## Example
**Worked by hand:** $[1,2,3]\cdot[4,5,6] = 1\cdot4 + 2\cdot5 + 3\cdot6 = 4 + 10 + 18 = 32$.

**Analogy:** A shopping basket `[2 apples, 3 breads]` dotted with prices `[$0.50, $2.00]` gives the total bill: 2·0.5 + 3·2 = $7. The dot product is a "weighted total".

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b), a @ b)          # 32 32

# A query scoring three keys (a mini attention step)
q = np.array([1.0, 0.0])
keys = np.array([[ 0.9, 0.1],       # similar direction
                 [ 0.0, 1.0],       # orthogonal
                 [-1.0, 0.0]])      # opposite
print(keys @ q)                     # [ 0.9  0.  -1. ]

cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(round(cos, 4))                # 0.9746  -> nearly the same direction
```

## Common Mistakes / Misconceptions
- **Assuming the dot product returns a vector.** It returns a scalar. Element-wise multiplication (`a * b`) returns a vector.
- **Treating the raw dot product as pure similarity.** It grows with vector length. Use cosine similarity (normalise first) when you only care about direction. This is also why attention divides by $\sqrt{d_k}$.
- **Dotting vectors of different lengths.** Both vectors must have the same dimension.

> **Note on overlap:** [Week 7 Q7 – What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md) builds on this. Cosine similarity is a *length-normalised* dot product, used for comparing embeddings in retrieval.

## Key Takeaways
- Dot product = multiply element-wise, then sum, giving one scalar.
- Geometrically it equals $\|a\|\|b\|\cos\theta$, so it measures alignment.
- Neurons, matrix multiplication, attention scores and logits are all built on dot products.
- Normalising the vectors turns the dot product into cosine similarity.

## Related Questions
- [What is a vector?](../Week-03-Mathematics-and-Neural-Network-Foundations/02-What-is-a-vector.md)
- [What is an attention score?](../Week-04-Transformer-Architecture/07-What-is-an-attention-score.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
