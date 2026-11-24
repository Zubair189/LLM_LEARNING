# What is an attention score?

> Week 04 – Transformer Architecture · Question 7

## Short Answer
An attention score is a number measuring how relevant one token (the key) is to another token (the query). In Transformers, it is the scaled dot product $s_{ij} = \mathbf{q}_i\cdot\mathbf{k}_j/\sqrt{d_k}$. Raw scores are unbounded. Applying softmax across each query's scores converts them into **attention weights** (non-negative, summing to 1), which decide how much of each value vector goes into the output.

## Detailed Explanation
**From scores to weights**
$$S = \frac{QK^T}{\sqrt{d_k}} \quad\text{(n × n raw scores)}\qquad A = \text{softmax}_{\text{row}}(S)\quad\text{(weights)}$$

- Row $i$ of $S$: how query token $i$ scores every key.
- Higher score means more aligned vectors, a larger weight, and more of that token's value in the output.

**Terminology** (sources differ, so be precise)
| Term | Range | Meaning |
|---|---|---|
| Raw score / logit | $(-\infty, \infty)$ | $q\cdot k$ (often scaled) |
| Attention weight / probability | [0, 1], rows sum to 1 | softmax of scores |
| Attention map / pattern | matrix | all weights, often visualised as a heatmap |

**Why scale by $\sqrt{d_k}$?** If the entries of $q$ and $k$ have variance 1, then $q\cdot k$ has variance $d_k$. With $d_k=128$, scores of ±20 or more are common, so softmax becomes nearly one-hot and gradients vanish. Dividing by $\sqrt{d_k}$ restores roughly unit variance.

**Modifications applied to scores before softmax**
- **Causal mask:** set future positions to $-\infty$ so their weight becomes 0.
- **Padding mask:** ignore pad tokens in batched inputs.
- **Position biases:** e.g. ALiBi adds a distance penalty. RoPE instead rotates Q and K so the score itself depends on relative position.

**Score function alternatives:** additive (Bahdanau) attention used a small MLP, $v^T\tanh(W[q;k])$. Dot-product attention won because it is a single fast matrix multiplication.

## Example
**Analogy:** A recruiter (query) rates each CV (key) for fit. The raw ratings are scores. Converting them into "percentage of interview time to spend on each candidate" is the softmax step, producing weights.

**Worked example** ($d_k = 2$):
$Q = \begin{bmatrix}1&2\\0&1\end{bmatrix}$, $K = \begin{bmatrix}1&0\\1&1\\0&2\end{bmatrix}$

- Raw $QK^T$: row 1: $[1\cdot1+2\cdot0,\ 1+2,\ 0+4] = [1, 3, 4]$; row 2: $[0, 1, 2]$
- Scaled by $1/\sqrt2$: $[0.707, 2.121, 2.828]$ and $[0, 0.707, 1.414]$
- Softmax per row gives the weights below.

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

Q = np.array([[1.0, 2.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])

S = Q @ K.T
print(S)                              # [[1. 3. 4.]
                                      #  [0. 1. 2.]]
S_scaled = S / np.sqrt(K.shape[1])
W = softmax(S_scaled)
print(W.round(3))                     # [[0.074 0.306 0.62 ]
                                      #  [0.14  0.284 0.576]]
print(W.sum(axis=1))                  # [1. 1.]
```
Both queries attend most to key 3, which is most aligned with them.

## Common Mistakes / Misconceptions
- **Using "score" and "weight" interchangeably.** Scores are pre-softmax. Weights are post-softmax.
- **Normalising over the wrong axis.** Softmax runs across *keys* (each row), not across queries.
- **Masking after softmax.** Set masked scores to $-\infty$ *before* softmax. Zeroing afterwards breaks the sum-to-1 property.
- **Over-interpreting heatmaps.** High weight is not proof of causal importance.

## Key Takeaways
- Attention score = scaled dot product of a query and a key.
- Softmax over each row turns the scores into attention weights that sum to 1.
- Scaling by $\sqrt{d_k}$ prevents saturated softmax and vanishing gradients.
- Masks and position schemes act on the scores before softmax.

## Related Questions
- [What is a dot product?](../Week-03-Mathematics-and-Neural-Network-Foundations/05-What-is-a-dot-product.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
