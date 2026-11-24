# What are Query, Key and Value?

> Week 04 – Transformer Architecture · Question 6

## Short Answer
Query, Key and Value are three different vectors computed from each token's embedding by multiplying it with three learned weight matrices ($W_Q, W_K, W_V$). The **query** is what this token is looking for. The **key** is what this token offers, used for matching. The **value** is the information this token passes on if selected. Attention scores come from query·key, and the output is a weighted sum of values. Separating these roles lets the model decide independently *how to match* and *what to transmit*.

## Detailed Explanation
$$Q = XW_Q \qquad K = XW_K \qquad V = XW_V$$
$$\text{output} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Vector | Question it answers | Used in |
|---|---|---|
| Query (Q) | "What am I looking for?" | scores: compared with every key |
| Key (K) | "What do I contain / how can I be found?" | scores: compared with every query |
| Value (V) | "What information do I hand over?" | weighted sum producing the output |

**Why three separate projections instead of using $X$ directly?**
- **Asymmetry:** "it" looking for its noun (query) is a different role from "animal" advertising itself as a noun (key). With $Q=K$, scores would be symmetric ($s_{ij}=s_{ji}$).
- **Matching ≠ content:** the features useful for *finding* a token (e.g. "I'm a plural subject") differ from the information worth *copying* (e.g. its meaning).
- **Learnable:** $W_Q, W_K, W_V$ are trained by backprop, so the model discovers which relationships matter.

**Shapes** (one head): $X$: `(n, d_model)`, $W_Q, W_K$: `(d_model, d_k)`, $W_V$: `(d_model, d_v)`. Scores $QK^T$: `(n, n)`. Output: `(n, d_v)`. Typically $d_k = d_v = d_{model}/h$ for $h$ heads.

**In self-attention** Q, K and V all come from the same sequence. **In cross-attention** Q comes from the decoder while K and V come from the encoder (or image features).

**Inference detail:** during generation, past tokens' K and V never change, so they are stored in the **KV cache**. Queries are only needed for the newest token. Grouped-query attention (GQA) shares K/V heads across several query heads to shrink that cache.

## Example
**Analogy: searching a video platform.**
- Your search text is the **query**.
- Each video's title, tags and description are its **key**.
- The video content itself is the **value**.

The platform scores query-vs-key matches, then shows you a ranked mix of content. Attention does the same thing, but returns a *weighted blend* of all values.

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

rng = np.random.default_rng(0)
X = rng.normal(size=(3, 4))                      # 3 tokens, d_model = 4
Wq, Wk, Wv = (rng.normal(size=(4, 2)) for _ in range(3))   # learned in practice

Q, K, V = X @ Wq, X @ Wk, X @ Wv                 # each (3, 2)
A = softmax(Q @ K.T / np.sqrt(2))                # (3, 3) attention weights
out = A @ V                                      # (3, 2)
print(Q.shape, K.shape, V.shape, A.shape, out.shape)
# (3, 2) (3, 2) (3, 2) (3, 3) (3, 2)
print(A.sum(axis=1))                             # [1. 1. 1.]  each row is a distribution
```

## Common Mistakes / Misconceptions
- **"Q, K and V are different inputs."** In self-attention they are three *projections of the same input*.
- **"Keys and values are the same thing."** Keys determine *how much* to attend. Values determine *what* gets passed on.
- **"The KV cache stores queries too."** Only keys and values are cached. A past token's query is never needed again.
- **Mixing up the formula order.** Scores are $QK^T$ (queries × keys), and then the weights multiply $V$.

## Key Takeaways
- Q, K and V are learned linear projections of token embeddings.
- Query·Key gives relevance scores. Softmax of the scores weights the Values.
- Separate roles let matching and information transfer be learned independently.
- K and V of past tokens are cached during generation (the KV cache).

## Related Questions
- [What is attention?](../Week-04-Transformer-Architecture/04-What-is-attention.md)
- [What is an attention score?](../Week-04-Transformer-Architecture/07-What-is-an-attention-score.md)
- [What is multi-head attention?](../Week-04-Transformer-Architecture/08-What-is-multi-head-attention.md)
- [What is the KV cache and why does it speed up generation? (bonus)](../Week-04-Transformer-Architecture/16-BONUS-What-is-the-KV-cache-and-why-does-it-speed-up-generation.md)
