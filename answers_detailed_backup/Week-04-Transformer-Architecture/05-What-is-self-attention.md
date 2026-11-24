# What is self-attention?

> Week 04 – Transformer Architecture · Question 5

## Short Answer
Self-attention is attention in which a sequence attends to itself. Every token produces a query, a key and a value from its own embedding, then computes how relevant every other token in the *same* sequence is to it and builds a new, context-aware representation as a weighted mix of their values. This is how a Transformer figures out that "it" refers to "the animal" or that "bank" means a riverbank, and it does this for all tokens in parallel.

## Detailed Explanation
**Formula** (for input matrix $X$ of shape `(n, d_model)`):
$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$
$$\text{SelfAttention}(X) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Shapes**
| Tensor | Shape | Meaning |
|---|---|---|
| $X$ | (n, d_model) | one embedding per token |
| $Q, K, V$ | (n, d_k) | projections |
| $QK^T$ | (n, n) | score of every token against every token |
| softmax(...) | (n, n) | each row sums to 1: "who should I listen to?" |
| output | (n, d_k) | a new context-aware vector per token |

**Self vs. other attention**
| Type | Q from | K, V from |
|---|---|---|
| Self-attention | sequence X | the same X |
| Cross-attention | decoder sequence | encoder output (or image features) |

**Bidirectional vs. causal**
- *Bidirectional* (BERT, encoders): every token sees all tokens, both left and right.
- *Causal / masked* (GPT, decoder-only LLMs): token $i$ sees only tokens $\le i$, so the model can be trained to predict the next token without cheating.

**Why it is powerful**
- **Contextual meaning:** the same word gets different representations in different sentences.
- **Direct long-range links:** token 1 and token 1,000 interact in one step.
- **Parallel:** all rows of the $n\times n$ matrix are computed at once.
- **Stacked:** dozens of layers let representations become increasingly abstract (syntax, then semantics, then task-level features).

## Example
**Analogy:** A group discussion where every participant listens to everyone else at once, weighting each person by how relevant they are to their own concern, then updates their view. Everyone does this simultaneously.

**Linguistic example:** "The animal didn't cross the street because **it** was too tired." When processing "it", self-attention places high weight on "animal". Change "tired" to "wide" and the weight shifts to "street".

**Numeric example** (for clarity, use $W_Q=W_K=W_V=I$, so $Q=K=V=X$):
```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

X = np.array([[1.0, 0.0],    # token A
              [0.0, 1.0],    # token B
              [1.0, 1.0]])   # token C (shares features with both)

S = X @ X.T / np.sqrt(2)     # (3,3) scores
A = softmax(S)               # attention weights, rows sum to 1
print(A.round(3))
# [[0.401 0.198 0.401]       A attends to itself and to C
#  [0.198 0.401 0.401]       B attends to itself and to C
#  [0.248 0.248 0.503]]      C attends mostly to itself
print((A @ X).round(3))      # new, context-mixed representations
# [[0.802 0.599]
#  [0.599 0.802]
#  [0.752 0.752]]
```
In a real model, learned $W_Q, W_K, W_V$ make the scores reflect *useful* relationships rather than raw similarity.

## Common Mistakes / Misconceptions
- **"Self-attention means a token attends only to itself."** It attends to *all* tokens in its own sequence, including itself.
- **Forgetting the causal mask in a decoder.** Without it, training leaks future tokens and the model learns nothing useful for generation.
- **"Self-attention knows word order."** It is permutation-equivariant. Order comes from positional encodings (e.g. RoPE).
- **Ignoring the $O(n^2)$ memory and compute** of the score matrix for long contexts.

> **Note on overlap:** [Q4 – What is attention?](../Week-04-Transformer-Architecture/04-What-is-attention.md) covers the general mechanism (any query over any set of keys). This question covers the specific case where Q, K and V all come from the same sequence.

## Key Takeaways
- Self-attention: each token's Q is compared against every token's K in the same sequence, and the result is used to mix their V's.
- It produces contextual representations and direct long-range connections.
- Encoders use bidirectional self-attention; GPT-style decoders use causal (masked) self-attention.
- It is computed for all tokens in parallel, at $O(n^2)$ cost.

## Related Questions
- [What is attention?](../Week-04-Transformer-Architecture/04-What-is-attention.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [What is multi-head attention?](../Week-04-Transformer-Architecture/08-What-is-multi-head-attention.md)
