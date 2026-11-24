# What is a scalar?

> Week 03 – Mathematics and Neural Network Foundations · Question 1

## Short Answer
A scalar is a single number, such as `3.5`, `-2`, or `0.001`. It has magnitude (size) but no direction and no dimensions. In deep learning, scalars show up everywhere as single values: a loss value, a learning rate, a bias term, a temperature setting, or one entry inside a larger vector or matrix.

## Detailed Explanation
Linear algebra organises numbers by how many indices you need to locate a value:

| Object | Indices needed | Shape (NumPy) | Example |
|---|---|---|---|
| Scalar | 0 | `()` | `7` |
| Vector | 1 | `(n,)` | `[1, 2, 3]` |
| Matrix | 2 | `(m, n)` | `[[1, 2], [3, 4]]` |
| Tensor | 3 or more | `(a, b, c, ...)` | a batch of images |

So a scalar is a **rank-0 tensor**: it has no axes.

Where scalars appear in LLM engineering:
- **Loss**: training reduces all prediction errors to one number, e.g. `loss = 2.31`. Gradients are computed with respect to this scalar.
- **Hyperparameters**: learning rate (`3e-4`), temperature (`0.7`), dropout probability (`0.1`), top-p (`0.9`).
- **Scaling factors**: in attention, scores are divided by the scalar $\sqrt{d_k}$.
- **Single weights and biases**: each parameter in a model is a scalar; a 7B model has about 7 billion of them.

**Scalar operations on arrays (broadcasting).** Multiplying a vector by a scalar scales every element: $c \cdot [x_1, x_2] = [c x_1, c x_2]$. This is how a learning rate scales a whole gradient in one step.

## Example
**Analogy:** A thermometer reading of 21 °C is a scalar: one number, no direction. Wind is not a scalar, because it has a speed *and* a direction (that is a vector).

```python
import numpy as np

lr = 0.01            # a scalar hyperparameter
temperature = 0.7    # another scalar

x = np.float32(3.5)
print(np.ndim(x), np.shape(x))          # 0 ()   -> zero axes

grad = np.array([1.0, 2.0, 3.0])
print(np.array([1.0, 2.0, 3.0]) * 2)    # [2. 4. 6.]  scalar broadcast to every element
print(lr * grad)                        # [0.01 0.02 0.03]  scaled gradient step
```

## Common Mistakes / Misconceptions
- **Treating a 1-element array as a scalar.** `np.array([3.5])` has shape `(1,)` (a vector of length 1), while `np.float32(3.5)` has shape `()`. Some library functions behave differently for the two. In PyTorch, `loss.backward()` needs a scalar (0-dim) tensor.
- **Thinking "scalar" means "small" or "unimportant".** One scalar such as the learning rate can decide whether training succeeds or diverges.
- **Confusing a scalar with a unit-free number.** Scalars can carry units (seconds, dollars). Only the dimensionality is zero.

## Key Takeaways
- A scalar is a single number: a rank-0 tensor with shape `()`.
- Loss values, hyperparameters and individual weights are all scalars.
- Multiplying a vector or matrix by a scalar scales every element (broadcasting).
- Scalars are the building block. Vectors, matrices and tensors are organised collections of them.

## Related Questions
- [What is a vector?](../Week-03-Mathematics-and-Neural-Network-Foundations/02-What-is-a-vector.md)
- [What is a matrix?](../Week-03-Mathematics-and-Neural-Network-Foundations/03-What-is-a-matrix.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
- [What is a learning rate? (bonus)](../Week-03-Mathematics-and-Neural-Network-Foundations/16-BONUS-What-is-a-learning-rate.md)
