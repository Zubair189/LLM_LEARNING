# What is a vector?

> Week 03 – Mathematics and Neural Network Foundations · Question 2

## Short Answer
A vector is an ordered list of numbers, such as `[3, 4]` or `[0.12, -0.5, 0.88, ...]`. You can read it in two ways: as a point or arrow in n-dimensional space (it has a length and a direction), or as a list of features describing one object. In LLMs, every token is represented as a vector (its embedding), and most model computation is vector arithmetic.

## Detailed Explanation
- **Notation:** $\mathbf{v} = [v_1, v_2, \dots, v_n]$, where $n$ is the *dimension*. In NumPy this is a 1-D array of shape `(n,)`.
- **Geometric view:** `[3, 4]` is an arrow from the origin to the point (3, 4). Its **length (L2 norm)** is $\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = 5$.
- **Feature view:** a house could be `[bedrooms, area_m2, age] = [3, 120, 15]`. Each position always means the same thing.

Core operations:

| Operation | Formula | Meaning |
|---|---|---|
| Addition | $\mathbf{a}+\mathbf{b} = [a_1+b_1, \dots]$ | combine / shift |
| Scalar multiply | $c\mathbf{a} = [c a_1, \dots]$ | stretch / shrink |
| Norm | $\|\mathbf{a}\| = \sqrt{\sum a_i^2}$ | length |
| Dot product | $\mathbf{a}\cdot\mathbf{b} = \sum a_i b_i$ | alignment / similarity |
| Normalise | $\mathbf{a}/\|\mathbf{a}\|$ | unit vector, keeps only the direction |

**Vectors in LLMs:**
- **Token embeddings**: each token maps to a vector, for example 4,096 numbers in a 7–8B model.
- **Hidden states**: the running representation of each token as it passes through the layers.
- **Logits**: a vector with one score per vocabulary entry, often more than 100k entries.
- **Gradients**: one number per parameter, telling you which way to adjust it.
- **Semantic search**: texts become vectors, and "similar meaning" becomes "nearby vectors" (Week 7).

```
   y
 4 |      * (3,4)
   |    /
   |  /   length = 5
   |/______ x
         3
```

## Example
**Analogy:** GPS coordinates `[latitude, longitude]` are a 2-D vector. A movie described by `[action, romance, comedy] = [0.9, 0.1, 0.4]` is a 3-D vector, and movies with similar vectors feel alike.

```python
import numpy as np

v = np.array([3.0, 4.0])
w = np.array([1.0, 2.0])

print(np.linalg.norm(v))        # 5.0          length
print(v + w, 2 * v)             # [4. 6.] [6. 8.]
print(v / np.linalg.norm(v))    # [0.6 0.8]    unit vector (direction only)
print(v @ w)                    # 11.0         dot product = 3*1 + 4*2
```

## Common Mistakes / Misconceptions
- **Confusing the vector's dimension with its length.** `[3, 4]` has dimension 2 and length 5.
- **Assuming positions are interchangeable.** Order matters: `[3, 120]` and `[120, 3]` are different houses.
- **Treating NumPy's 1-D arrays as row or column vectors.** A shape `(n,)` array is neither. Use `(1, n)` or `(n, 1)` when orientation matters.
- **Reading meaning into single embedding dimensions.** Individual dimensions of learned embeddings are rarely interpretable. Meaning lives in directions and distances.

## Key Takeaways
- A vector is an ordered list of numbers, which you can also see as a point or arrow in space.
- Its norm is its length, and its direction carries meaning (for example, the meaning of a token).
- Everything an LLM processes (tokens, hidden states, logits, gradients) is held in vectors.
- The dot product of two vectors measures how aligned they are. Attention and similarity search are built on it.

> **Note on overlap:** [Week 7 Q3 – What is a vector representation?](../Week-07-Embeddings-and-Vector-Databases/03-What-is-a-vector-representation.md) asks how *text* is encoded as vectors. This question covers the mathematical object itself.

## Related Questions
- [What is a scalar?](../Week-03-Mathematics-and-Neural-Network-Foundations/01-What-is-a-scalar.md)
- [What is a dot product?](../Week-03-Mathematics-and-Neural-Network-Foundations/05-What-is-a-dot-product.md)
- [What are token embeddings?](../Week-04-Transformer-Architecture/10-What-are-token-embeddings.md)
- [What is a vector representation?](../Week-07-Embeddings-and-Vector-Databases/03-What-is-a-vector-representation.md)
