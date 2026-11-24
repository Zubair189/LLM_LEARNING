# What is masked self-attention?

> Week 04 – Transformer Architecture · Question 13

## Short Answer
Masked (causal) self-attention is self-attention in which each token may attend only to itself and earlier tokens, never to later ones. It is implemented by setting the attention scores for future positions to $-\infty$ before softmax, so their weights become exactly 0. This lets a decoder be trained on next-token prediction for every position of a sequence in parallel without "seeing the answer", and it matches how text is generated: left to right.

## Detailed Explanation
**The mask**
$$M_{ij} = \begin{cases}0 & j \le i \\ -\infty & j > i\end{cases}\qquad A = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right)$$

```
            keys →  t1   t2   t3   t4
queries  t1       [ ✓    ✗    ✗    ✗ ]
   ↓     t2       [ ✓    ✓    ✗    ✗ ]
         t3       [ ✓    ✓    ✓    ✗ ]
         t4       [ ✓    ✓    ✓    ✓ ]     ✓ allowed, ✗ masked (-inf)
```
The allowed region is the lower triangle, hence "causal" or "triangular" mask.

**Why it is needed**
- In training, the whole target sentence is fed in at once (teacher forcing). Without the mask, the position predicting token 3 could just look at token 3, so the loss would be trivially low and the model would learn nothing usable for generation.
- The mask makes training conditions match inference, where future tokens don't exist yet.

**Why $-\infty$ and not 0?** Masking happens on *scores* before softmax: $e^{-\infty} = 0$, so the masked weights are exactly 0 and the remaining weights still sum to 1. Setting the scores to 0 would still give $e^0 = 1$, i.e. non-zero attention to the future. (Implementations often use a large negative number such as `-1e9`, or the dtype minimum.)

**Other masks you'll meet**
| Mask | Purpose |
|---|---|
| Causal mask | no attending to future tokens (decoders) |
| Padding mask | ignore pad tokens when batching variable-length inputs |
| Sliding-window mask | attend only to the last $w$ tokens (long-context efficiency, e.g. some Mistral and Gemma layers) |
| Prefix-LM mask | bidirectional over the prompt, causal over the response |
| Document mask | prevent attention across packed, unrelated documents in a training batch |

**"Masked self-attention" ≠ "masked language modelling".** BERT's MLM masks *input tokens* (replaces them with `[MASK]`) and uses *bidirectional* attention. Causal masking masks *attention connections*.

## Example
**Analogy:** Taking an exam where questions are revealed one at a time. You can review earlier questions and your answers, but the next page is sealed. Masking is the seal.

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

S = np.array([[2.0, 1, 0, 1],
              [1, 2, 1, 0],
              [0, 1, 2, 1],
              [1, 0, 1, 2]])                         # raw scores for 4 tokens
n = S.shape[0]
future = np.triu(np.ones((n, n), dtype=bool), k=1)   # True above the diagonal
A = softmax(np.where(future, -np.inf, S))
print(A.round(3))
# [[1.    0.    0.    0.   ]     token 1 can only attend to itself
#  [0.269 0.731 0.    0.   ]
#  [0.09  0.245 0.665 0.   ]
#  [0.197 0.072 0.197 0.534]]    token 4 sees all four
print(A.sum(axis=1))             # [1. 1. 1. 1.]  rows still sum to 1
```
Row 2 by hand: softmax([1, 2]) = [e/(e+e²), e²/(e+e²)] = [0.269, 0.731].

## Common Mistakes / Misconceptions
- **Applying the mask after softmax** (zeroing weights). The rows then no longer sum to 1, and gradients are wrong.
- **Using a strict lower triangle** (`k=0` in `triu`), which also blocks a token from attending to itself.
- **Confusing causal masking with BERT-style token masking.**
- **Forgetting the padding mask** in batched inference, so pad tokens leak into attention.

## Key Takeaways
- The causal mask blocks attention to future positions by adding $-\infty$ to their scores before softmax.
- It enables parallel next-token training without information leakage.
- It is the defining feature of decoder / GPT-style models.
- Padding, sliding-window and document masks use the same mechanism for other purposes.

## Related Questions
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [What is a decoder?](../Week-04-Transformer-Architecture/12-What-is-a-decoder.md)
- [What is an attention score?](../Week-04-Transformer-Architecture/07-What-is-an-attention-score.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
