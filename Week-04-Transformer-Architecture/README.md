# Week 04 – Transformer Architecture

This week explains the architecture behind every modern LLM. It starts with the problems of recurrent networks (vanishing gradients, sequential computation, the fixed-size bottleneck) and shows how the Transformer solves them with attention. It then works through the components one at a time: query/key/value projections, attention scores and softmax weights, multiple heads, token embeddings, and positional information (sinusoidal and RoPE). It then assembles them into encoders, decoders and the decoder-only GPT design that dominates today, including causal masking and the KV cache that makes generation fast. By the end you should be able to sketch a decoder-only Transformer from memory and implement its attention layer in NumPy.

## Questions
1. [What problems do traditional RNNs have?](01-What-problems-do-traditional-RNNs-have.md)
2. [What is a Transformer?](02-What-is-a-Transformer.md)
3. [Why did Transformers become important?](03-Why-did-Transformers-become-important.md)
4. [What is attention?](04-What-is-attention.md)
5. [What is self-attention?](05-What-is-self-attention.md)
6. [What are Query, Key and Value?](06-What-are-Query-Key-and-Value.md)
7. [What is an attention score?](07-What-is-an-attention-score.md)
8. [What is multi-head attention?](08-What-is-multi-head-attention.md)
9. [What are positional encodings?](09-What-are-positional-encodings.md)
10. [What are token embeddings?](10-What-are-token-embeddings.md)
11. [What is an encoder?](11-What-is-an-encoder.md)
12. [What is a decoder?](12-What-is-a-decoder.md)
13. [What is masked self-attention?](13-What-is-masked-self-attention.md)
14. [How does GPT use the Transformer architecture?](14-How-does-GPT-use-the-Transformer-architecture.md)
15. [Why are Transformers effective at processing language?](15-Why-are-Transformers-effective-at-processing-language.md)
16. **Bonus:** [What is the KV cache and why does it speed up generation?](16-BONUS-What-is-the-KV-cache-and-why-does-it-speed-up-generation.md)

**Overlap guide:** Q2, Q3 and Q15 are deliberately related. Q2 asks *what* a Transformer is, Q3 asks *why it mattered historically*, and Q15 asks *why the mechanism suits language*. Q4 (attention in general) and Q5 (self-attention) are also related. Week 12 Q2 revisits the whole architecture for revision.

## Practical Exercise (1–2 hours): causal multi-head attention in NumPy
1. **Scaled dot-product attention:** implement `attention(Q, K, V, mask=None)` returning `softmax(QK^T/√d_k + mask) @ V`. Use a numerically stable softmax. Verify that every row of the weights sums to 1.
2. **Causal mask:** build it with `np.triu(..., k=1)` and `-np.inf`. Test it: change the *last* token's embedding and confirm that the outputs for earlier positions don't change.
3. **Multi-head:** with `d_model=16`, `h=4`, write `split_heads`/`merge_heads` using `reshape` + `transpose`, and add an output projection `W_O`.
4. **Positions:** add RoPE to Q and K. Show that the score between a query at position m and a key at position n depends only on m−n.
5. **KV cache:** generate 10 steps two ways (full recompute vs. cache) and assert the outputs match with `np.allclose`. Count the K/V projections each way.
6. **Stretch:** add a feed-forward block, residual connections and RMSNorm to make one full pre-norm decoder block. Then compare your attention against `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)` if PyTorch is available.

Success criterion: all assertions pass, and you can explain every tensor shape in your code.

## Explain Aloud (no notes)
1. **Attention end-to-end.** Cover Q/K/V projections, the scaled dot product, softmax, and the weighted sum of values. Explain why we scale by √d_k and why there are multiple heads.
2. **How GPT generates a reply.** Cover the decoder-only stack, causal masking, next-token prediction training, prefill vs. decode, and why the KV cache makes decoding fast (and what it costs in memory).
3. **Why Transformers replaced RNNs.** Cover vanishing gradients, sequential vs. parallel computation, direct long-range links, and why attention needs positional information (sinusoidal vs. RoPE).
