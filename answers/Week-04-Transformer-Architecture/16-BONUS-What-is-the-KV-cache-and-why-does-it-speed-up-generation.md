# What is the KV cache and why does it speed up generation?

> Week 04 – Transformer Architecture · Bonus question (added)

## In One Line
The KV cache **saves the keys and values of earlier tokens** so the model doesn't recompute them every time it writes a new token.

## Simple Explanation
An LLM writes one token at a time. For each new token, attention needs the **keys** (K) and **values** (V) of **all** the earlier tokens.

**Without a cache:** at every step, the model recomputes K and V for the whole text so far. Step 1 handles 1 token, step 2 handles 2, step 100 handles 100… Lots of repeated work.

**Key fact:** because of the causal mask, an earlier token never sees later tokens. So its K and V **never change**. Recomputing them is pure waste.

**With a cache:**
1. **Prefill:** read the prompt once, in parallel, and store K and V for every token in every layer.
2. **Decode:** for each new token, compute only **its** query, key and value, add its K and V to the cache, and attend over everything in the cache.

**Why only K and V, not Q?** A token's query is only needed to compute that token's own output, which is already done. Later tokens need earlier tokens' keys (to score them) and values (to read them).

**The cost: memory.** The cache grows with every token, every layer and every user being served. For a Llama-3-8B-sized model, one 8,192-token conversation needs about **1 GiB** of GPU memory just for the cache. That is why tricks like **grouped-query attention** (sharing K/V between heads) and **prompt caching** exist.

**Link to LLMs:** every LLM service uses a KV cache. It is a big reason chatbots can reply quickly.

## Real-Life Example
Taking minutes in a long meeting. Without notes, every time someone speaks you would replay the whole recording from the start. With minutes (the cache), you just add one line for the newest remark and glance at your notes.

## Code Example
Count how many K/V calculations are needed to generate 1,000 tokens:

```python
n = 1000                                   # tokens to generate
no_cache = sum(range(1, n + 1))            # step t recomputes K and V for all t tokens
with_cache = n                             # step t computes K and V for 1 new token
print(no_cache, with_cache)                # 500500 1000

# Cache size: 2 (K and V) x 32 layers x 8 KV heads x 128 numbers x 8192 tokens x 2 bytes
gib = 2 * 32 * 8 * 128 * 8192 * 2 / 2**30
print(gib, "GiB")                          # 1.0 GiB
```

**What happened:** the cache cut the K/V work from about half a million calculations to 1,000. The second part shows the price: 1 GiB of memory for one long conversation.

## Watch Out
- The cache does not make attention free: each new token still reads the **whole** cache. It only removes the recomputation.
- It is used when **generating**, not during training (training processes whole sequences in parallel).

## Remember
- Cache past tokens' **K and V**, never Q.
- Prefill once, then decode one token at a time.
- Faster generation, paid for with memory.

## Related Questions
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [What is multi-head attention?](../Week-04-Transformer-Architecture/08-What-is-multi-head-attention.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
