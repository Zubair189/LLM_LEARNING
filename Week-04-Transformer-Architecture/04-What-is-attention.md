# What is attention?

> Week 04 – Transformer Architecture · Question 4

## Short Answer
Attention is a mechanism that lets a model focus on the most relevant parts of its input when computing each output. For a given query, it scores every available item (key) for relevance, turns those scores into weights with softmax, and returns a weighted average of the items' content (values). It is a learned, differentiable "soft lookup", and the core operation of Transformers: $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})\,V$.

## Detailed Explanation
**The three steps**
1. **Score:** compare the query with each key, $s_i = \mathbf{q}\cdot\mathbf{k}_i / \sqrt{d_k}$.
2. **Normalise:** $\alpha_i = \text{softmax}(s)_i$, so the weights are positive and sum to 1.
3. **Aggregate:** output $= \sum_i \alpha_i \mathbf{v}_i$.

```
query ─┬─ · k1 → s1 ┐
       ├─ · k2 → s2 ├─ softmax → α1, α2, α3 → α1·v1 + α2·v2 + α3·v3 = output
       └─ · k3 → s3 ┘
```

**A soft dictionary lookup**
| | Python dict | Attention |
|---|---|---|
| Lookup | exact key match | similarity to *all* keys |
| Result | one value | weighted blend of all values |
| Learnable? | no | yes, gradients flow through everything |

**Why divide by $\sqrt{d_k}$?** Dot products of random $d_k$-dimensional vectors have variance about $d_k$. Large scores push softmax into near one-hot regions with tiny gradients. Scaling keeps scores in a healthy range.

**History and types**
- **Bahdanau/Luong attention (2014–15):** a decoder RNN attends over encoder states in translation, fixing the fixed-vector bottleneck.
- **Self-attention:** queries, keys and values all come from the *same* sequence (the Transformer's key idea).
- **Cross-attention:** queries from one sequence, keys/values from another (decoder attending to encoder, text attending to image features).
- **Causal (masked) attention:** tokens can only attend to earlier positions (GPT-style).

**Cost:** for $n$ tokens, computing all scores is $O(n^2)$. Long-context efficiency tricks (FlashAttention, sliding windows, GQA, sparse attention) target this.

## Example
**Analogy:** In a library, your question is the **query**. Each book's spine label is a **key**. The book's content is the **value**. Instead of borrowing one book, you skim all of them in proportion to how well their labels match your question, and blend what you read.

**Worked example:** query $q=[1,0]$; keys $k_1=[1,0]$, $k_2=[0,1]$, $k_3=[0.7,0.7]$; $d_k=2$.
- Scores: $[1, 0, 0.7]/\sqrt2 = [0.707, 0, 0.495]$
- Softmax gives weights $[0.434, 0.214, 0.351]$
- Output $= 0.434\cdot[10,0] + 0.214\cdot[0,10] + 0.351\cdot[5,5] = [6.10, 3.90]$

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()

q = np.array([1.0, 0.0])
K = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
V = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])

scores = K @ q / np.sqrt(2)
weights = softmax(scores)
print(scores.round(3))           # [0.707 0.    0.495]
print(weights.round(3))          # [0.434 0.214 0.351]
print((weights @ V).round(3))    # [6.101 3.899]  leans toward v1, the best match
```

## Common Mistakes / Misconceptions
- **"Attention picks one item."** It produces a *weighted blend*. Hard selection would not be differentiable.
- **"Attention weights explain the model's reasoning."** They are a useful signal but not a faithful explanation on their own.
- **Treating attention and self-attention as synonyms.** Self-attention is the special case where Q, K and V come from the same sequence.
- **Forgetting the $\sqrt{d_k}$ scaling** when implementing it.

## Key Takeaways
- Attention = score (query·key), normalise (softmax), aggregate (weighted sum of values).
- It is a differentiable soft lookup that lets each output focus on relevant inputs.
- Self-, cross- and causal attention are variants of the same formula.
- Cost grows quadratically with sequence length.

## Related Questions
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is an attention score?](../Week-04-Transformer-Architecture/07-What-is-an-attention-score.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
