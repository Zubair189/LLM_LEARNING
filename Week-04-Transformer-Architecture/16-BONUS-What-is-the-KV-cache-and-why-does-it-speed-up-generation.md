# What is the KV cache and why does it speed up generation?

> Week 04 – Transformer Architecture · Bonus question (added)

## Short Answer
During autoregressive generation, each new token must attend to the keys and values of all previous tokens. Those past keys and values never change, because of the causal mask. The KV cache stores them in GPU memory after they are first computed, so each new step only computes Q, K and V for the *one* new token and reuses the rest. This turns repeated full-prefix recomputation into a small incremental update, making generation dramatically faster. The cost is memory that grows with sequence length, layers and batch size.

## Detailed Explanation
**Without a cache:** to generate token $t$, rerun the model on all $t$ tokens. Total work over a generation of $n$ tokens is $1+2+\dots+n = O(n^2)$ token-passes through every layer.

**With a cache:**
1. **Prefill:** process the prompt in one parallel pass, and store K and V for every layer and every prompt token.
2. **Decode:** for each new token, compute its q, k and v only, append k and v to the cache, and attend $q$ against all cached K, V.

Only $O(1)$ new tokens go through the projections and MLPs per step. Attention still reads all $t$ cached entries, which is cheap arithmetic but memory-bandwidth heavy.

```
step t:   new token ─► q_t, k_t, v_t
                        │     └──append──► [K cache: k_1 … k_t]  [V cache: v_1 … v_t]
                        └── attend over full cache ──► output ─► next token
```

**Why only K and V (not Q)?** A past token's query was only needed to compute *that* token's output, which is already done. Future tokens need past tokens' keys (to score them) and values (to read them).

**Why it's valid:** with causal masking, token $i$'s representation in every layer depends only on tokens $\le i$, so its K and V are identical at every later step.

**Memory cost**
$$\text{bytes} = 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{seq\_len} \times \text{batch} \times \text{bytes per value}$$
(the 2 is for K and V)

For a Llama-3-8B-like model (32 layers, 8 KV heads via GQA, head_dim 128, fp16), one 8,192-token sequence needs **1 GiB**. With 32 full KV heads (no GQA) it would be 4 GiB. For long contexts and large batches, the KV cache, not the weights, often limits throughput.

**Techniques built around the KV cache**
| Technique | Idea |
|---|---|
| GQA / MQA / MLA | fewer or compressed KV heads → smaller cache |
| PagedAttention (vLLM) | store the cache in pages like virtual memory → less fragmentation, more concurrent requests |
| Prefix / prompt caching | reuse the cache for a shared system prompt or documents across requests (offered by major LLM APIs at reduced price) |
| KV-cache quantisation | store K and V in 8-bit or lower |
| Sliding-window attention / eviction | keep only recent or important entries |

## Example
**Analogy:** Taking minutes in a long meeting. Without notes, every time someone speaks you'd replay the entire recording from the start to recall context. With minutes (the cache), you just add a line for the newest remark and glance over your notes.

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()

rng = np.random.default_rng(0)
d = 4
Wq, Wk, Wv = (rng.normal(size=(d, d)) for _ in range(3))
tokens = rng.normal(size=(6, d))           # 6 tokens arriving one at a time

def attend(q, K, V):
    return softmax(K @ q / np.sqrt(d)) @ V

# No cache: recompute K, V for the whole prefix every step
out_nc, rows_nc = [], 0
for t in range(len(tokens)):
    prefix = tokens[: t + 1]
    K, V = prefix @ Wk, prefix @ Wv
    rows_nc += t + 1
    out_nc.append(attend(tokens[t] @ Wq, K, V))

# With cache: compute K, V for the new token only, append to the cache
K_c, V_c, out_c, rows_c = np.empty((0, d)), np.empty((0, d)), [], 0
for t in range(len(tokens)):
    x = tokens[t]
    K_c, V_c = np.vstack([K_c, x @ Wk]), np.vstack([V_c, x @ Wv])
    rows_c += 1
    out_c.append(attend(x @ Wq, K_c, V_c))

print(np.allclose(out_nc, out_c))   # True  identical results
print(rows_nc, rows_c)              # 21 6  K/V projections computed: O(n^2) vs O(n)

# Memory for one 8k-token sequence, Llama-3-8B-like config, fp16
print(2 * 32 * 8 * 128 * 8192 * 2 / 2**30, "GiB")   # 1.0 GiB
```

## Common Mistakes / Misconceptions
- **"The KV cache makes attention O(1)."** Each step still attends over all $t$ cached tokens. It removes the redundant *recomputation*, not the attention reads.
- **"It helps training."** Training processes full sequences in parallel and doesn't need it. It is an inference optimisation.
- **"Caching is free."** Memory grows linearly with context and batch size and often caps how many users a GPU can serve.
- **Caching queries.** Only keys and values are cached.

## Key Takeaways
- The KV cache stores past tokens' keys and values per layer so they are computed only once.
- Generation splits into a parallel *prefill* and a step-by-step *decode* that reuses the cache.
- It turns quadratic recomputation into incremental work, a large speed-up.
- Its memory footprint motivates GQA/MLA, PagedAttention, KV quantisation and prompt caching.

## Related Questions
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [What is multi-head attention?](../Week-04-Transformer-Architecture/08-What-is-multi-head-attention.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
