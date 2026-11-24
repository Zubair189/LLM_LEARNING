# Why do neural networks use matrices?

> Week 03 – Mathematics and Neural Network Foundations · Question 4

## Short Answer
A neural-network layer computes many weighted sums at once: every output neuron combines every input. A matrix multiplication $XW + b$ expresses exactly that, for a whole batch of examples, in one operation. Matrices give a compact notation for the computation, clean rules for computing gradients, and, most importantly, a computation that GPUs and TPUs are built to run extremely fast in parallel.

## Detailed Explanation
**1. One neuron = one dot product.** A neuron computes $z = w_1x_1 + w_2x_2 + \dots + b = \mathbf{w}\cdot\mathbf{x} + b$.

**2. One layer = many neurons = a matrix.** Stack each neuron's weight vector as a column of $W$:

$$\mathbf{h} = \mathbf{x}W + \mathbf{b}, \quad \mathbf{x}\in\mathbb{R}^{d_{in}},\; W\in\mathbb{R}^{d_{in}\times d_{out}}$$

**3. A batch = many inputs = another matrix.** Stack $B$ inputs as rows of $X$ (shape `(B, d_in)`), and $XW$ computes every neuron for every example together.

```
X (batch × in)     W (in × out)        H (batch × out)
[ x1 ]             [ w_neuron1 ... ]   [ h1 ]
[ x2 ]      @      [   columns     ] = [ h2 ]
[ x3 ]                                 [ h3 ]
```

**Why this matters**
| Reason | Explanation |
|---|---|
| Compact | One line `X @ W + b` replaces nested loops over examples, neurons and inputs |
| Fast | GPUs and TPUs have dedicated matrix-multiply hardware (tensor cores). Loops in Python would be thousands of times slower |
| Differentiable | Gradients have clean matrix forms, e.g. $\partial L/\partial W = X^T \, \partial L/\partial H$ |
| Composable | Layers stack as a chain of matrix products with non-linearities between them |
| Scalable | The same code handles 10 or 10,000 neurons. Only the shapes change |

**In Transformers:** Q/K/V projections, the attention scores $QK^T$, the attention-weighted sum $AV$, the output projection and the feed-forward MLP are all matrix multiplications. They account for the large majority of an LLM's FLOPs, which is why "how fast can we multiply matrices" largely decides training and inference cost.

**Why non-linearities are still needed:** a product of matrices is just another matrix ($W_2(W_1x) = (W_2W_1)x$). Without activation functions between layers, a deep network would collapse into a single linear layer.

## Example
**Analogy:** A restaurant chain prices 3 orders (rows) across 2 items (burger, fries) using a price matrix. Instead of computing each bill by hand, one matrix multiplication gives every bill at once. Change the menu, and you change one matrix.

```python
import numpy as np

X = np.array([[1.0,  2.0],     # 3 examples, 2 features each
              [0.5, -1.0],
              [3.0,  0.0]])
W = np.array([[0.1, 0.2, 0.3, 0.4],   # layer: 2 inputs -> 4 neurons
              [0.5, 0.6, 0.7, 0.8]])
b = np.zeros(4)

H = X @ W + b                  # all 3 examples x 4 neurons in one call
print(H.shape)                 # (3, 4)
print(H)
# [[ 1.1   1.4   1.7   2.  ]
#  [-0.45 -0.5  -0.55 -0.6 ]
#  [ 0.3   0.6   0.9   1.2 ]]
```
Hand check for H[0,0]: 1·0.1 + 2·0.5 = 1.1.

## Common Mistakes / Misconceptions
- **Thinking matrices are only a notation trick.** They are the reason deep learning runs at practical speed on modern hardware.
- **Forgetting that stacked linear layers collapse.** Depth only helps if non-linear activations sit between the matrix multiplications.
- **Writing Python loops over neurons or examples.** Always vectorise. Batching is "free" parallelism.
- **Mixing up the batch and feature axes.** By convention, rows are examples and columns are features.

## Key Takeaways
- A layer is a set of weighted sums, which is exactly what a matrix multiplication computes.
- Batching stacks inputs into a matrix, so one multiplication processes many examples.
- GPUs are optimised for matrix multiplication, so the matrix form is what makes training and serving LLMs feasible.
- Activation functions between matrix multiplications give the network its expressive power.

## Related Questions
- [What is a matrix?](../Week-03-Mathematics-and-Neural-Network-Foundations/03-What-is-a-matrix.md)
- [What is forward propagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/11-What-is-forward-propagation.md)
- [What is an activation function?](../Week-03-Mathematics-and-Neural-Network-Foundations/08-What-is-an-activation-function.md)
- [What is a neural network?](../Week-01-Introduction-to-Artificial-Intelligence/08-What-is-a-neural-network.md)
