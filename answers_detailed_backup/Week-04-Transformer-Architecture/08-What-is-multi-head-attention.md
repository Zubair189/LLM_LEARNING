# What is multi-head attention?

> Week 04 – Transformer Architecture · Question 8

## Short Answer
Multi-head attention runs several attention operations ("heads") in parallel, each with its own learned Q, K and V projections in a smaller subspace. Their outputs are concatenated and mixed by an output projection. Different heads can specialise in different relationships, for example one tracking syntax, another coreference, another the previous token. That gives richer representations than a single attention at the same total cost.

## Detailed Explanation
$$\text{head}_i = \text{Attention}(XW_Q^{(i)},\, XW_K^{(i)},\, XW_V^{(i)})$$
$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O$$

With $h$ heads, each head has dimension $d_k = d_{model}/h$, so total compute and parameters are about the same as one full-width head.

```
           ┌─ head 1 (d_k) ─┐
X (n×d) ───┼─ head 2 (d_k) ─┼── concat (n×d) ── W_O ── output (n×d)
           └─ head h (d_k) ─┘
```

**Why multiple heads?**
- **One softmax = one weighting.** A single head must average all relationships into one distribution per token. Multiple heads can attend to different tokens for different reasons at the same time.
- **Specialisation (observed empirically):** previous-token heads, induction heads (copy patterns seen earlier, a key mechanism for in-context learning), syntactic heads (verb to subject), and so on.
- **Subspaces:** each head works in its own learned projection, focusing on a subset of features.

**Typical configurations**
| Model | d_model | heads | d_head |
|---|---|---|---|
| Original Transformer (base) | 512 | 8 | 64 |
| GPT-2 small | 768 | 12 | 64 |
| Llama-3-8B-class | 4096 | 32 query / 8 KV | 128 |

**Efficient variants (important for inference)**
- **Multi-Query Attention (MQA):** all query heads share one K/V head, giving a tiny KV cache.
- **Grouped-Query Attention (GQA):** groups of query heads share K/V heads (e.g. 32 Q heads, 8 KV heads). This is the common default in modern open LLMs.
- **Multi-head Latent Attention (MLA, DeepSeek):** compresses K/V into a low-rank latent to shrink the cache further.

## Example
**Analogy:** A panel of specialists reviewing the same document: a grammarian, a fact-checker, a style editor and a legal reviewer. Each highlights different connections, and their notes are combined into a single report ($W_O$).

**Implementation with reshapes** (the standard trick: one big projection, then split into heads):
```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

rng = np.random.default_rng(1)
n, d_model, h = 4, 8, 2
d_k = d_model // h                                   # 4
X = rng.normal(size=(n, d_model))
Wq, Wk, Wv, Wo = (rng.normal(size=(d_model, d_model)) * 0.3 for _ in range(4))

def split(M):                    # (n, d_model) -> (h, n, d_k)
    return M.reshape(n, h, d_k).transpose(1, 0, 2)

Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
A = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(d_k))  # (h, n, n): one attention map per head
heads = A @ V                                          # (h, n, d_k)
concat = heads.transpose(1, 0, 2).reshape(n, d_model)  # (n, d_model)
out = concat @ Wo
print(Q.shape, A.shape, heads.shape, out.shape)
# (2, 4, 4) (2, 4, 4) (2, 4, 4) (4, 8)
```
Each of the 2 heads has its own 4×4 attention map, so it can focus on different tokens.

## Common Mistakes / Misconceptions
- **"More heads = more parameters."** Heads split $d_{model}$. The parameter count stays roughly constant.
- **"Each head is a separate full-size attention."** Each works in a $d_{model}/h$ subspace.
- **"Every head learns something meaningful."** Studies show many heads can be pruned with little loss. Specialisation is emergent, not guaranteed.
- **Forgetting $W_O$.** Without it, heads cannot interact before the next layer.

## Key Takeaways
- Multi-head attention = $h$ parallel attentions in smaller subspaces, concatenated and projected.
- Heads can capture different relationships simultaneously.
- Total cost is about the same as single-head attention of full width.
- GQA, MQA and MLA share or compress K/V across heads to cut KV-cache memory at inference.

## Related Questions
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is the KV cache and why does it speed up generation? (bonus)](../Week-04-Transformer-Architecture/16-BONUS-What-is-the-KV-cache-and-why-does-it-speed-up-generation.md)
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
