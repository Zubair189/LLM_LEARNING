# What is softmax?

> Week 03 – Mathematics and Neural Network Foundations · Question 10

## Short Answer
Softmax turns a vector of arbitrary real numbers (logits) into a probability distribution: every output is between 0 and 1 and they sum to 1. It exponentiates each value and divides by the sum of exponentials, $\text{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}$, so larger logits get disproportionately more probability. In LLMs, softmax produces the next-token distribution and the attention weights.

## Detailed Explanation
$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Why exponentiate?**
- $e^z > 0$ always, so negative logits still get positive probability.
- It amplifies differences: a logit gap of 1 means a probability ratio of $e \approx 2.72$.
- It is smooth and differentiable, and pairs neatly with cross-entropy loss (the combined gradient is simply $\hat{p} - y$).

**Properties**
- **Order-preserving:** the largest logit always gets the largest probability.
- **Shift-invariant:** $\text{softmax}(z + c) = \text{softmax}(z)$. Implementations therefore subtract $\max(z)$ first to avoid overflow, the "numerically stable softmax".
- **"Soft" argmax:** it is a differentiable approximation of picking the maximum.

**Temperature:** $\text{softmax}(z/T)$
| T | Effect | Use |
|---|---|---|
| → 0 | approaches one-hot (argmax) | deterministic answers |
| 1 | the model's raw distribution | default |
| > 1 | flatter, more random | creative generation |

**Where it appears in Transformers**
1. **Output layer:** logits over the vocabulary become next-token probabilities.
2. **Attention:** each row of $QK^T/\sqrt{d_k}$ goes through softmax to get attention weights that sum to 1.

## Example
**Worked by hand for z = [2, 1, 0.1]:**
1. Exponentiate: $e^2 = 7.389$, $e^1 = 2.718$, $e^{0.1} = 1.105$
2. Sum: $7.389 + 2.718 + 1.105 = 11.213$
3. Divide: $[7.389/11.213,\ 2.718/11.213,\ 1.105/11.213] = [0.659, 0.242, 0.099]$

**Analogy:** Votes in an election where enthusiasm counts exponentially. A slightly more popular candidate wins a clearly larger vote share, but everyone gets some share.

```python
import numpy as np

def softmax(z):
    e = np.exp(z - np.max(z))     # subtract max for numerical stability
    return e / e.sum()

z = np.array([2.0, 1.0, 0.1])
p = softmax(z)
print(p.round(3), p.sum())        # [0.659 0.242 0.099] 1.0

print(softmax(np.array([1000.0, 999.0])).round(4))   # [0.7311 0.2689]  (naive exp would overflow)

for T in [0.5, 1.0, 2.0]:
    print(T, softmax(z / T).round(3))
# 0.5 [0.864 0.117 0.019]   sharper
# 1.0 [0.659 0.242 0.099]
# 2.0 [0.502 0.304 0.194]   flatter
```

## Common Mistakes / Misconceptions
- **Naive implementation** `np.exp(z) / np.exp(z).sum()` overflows for large logits. Always subtract the max.
- **Applying softmax twice** (e.g. in the model and again in the loss). PyTorch's `CrossEntropyLoss` expects raw logits.
- **Reading softmax outputs as calibrated confidence.** A 0.95 probability does not mean the model is right 95% of the time.
- **Confusing softmax with sigmoid.** Sigmoid scores each output independently (multi-label). Softmax makes outputs compete (exactly one class).

## Key Takeaways
- Softmax maps logits to a valid probability distribution.
- Exponentiation makes outputs positive and amplifies differences between logits.
- Subtract the max before exponentiating for stability. Divide by a temperature to sharpen or flatten.
- It is used for both next-token prediction and attention weights.

## Related Questions
- [What is a probability distribution?](../Week-03-Mathematics-and-Neural-Network-Foundations/07-What-is-a-probability-distribution.md)
- [What is an activation function?](../Week-03-Mathematics-and-Neural-Network-Foundations/08-What-is-an-activation-function.md)
- [What is an attention score?](../Week-04-Transformer-Architecture/07-What-is-an-attention-score.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
