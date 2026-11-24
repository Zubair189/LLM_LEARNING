# What is a Transformer?

> Week 04 – Transformer Architecture · Question 2

## Short Answer
A Transformer is a neural-network architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), that processes sequences using self-attention instead of recurrence. Each layer lets every token gather information from every other relevant token, then refines it with a feed-forward network, with residual connections and normalisation around both. It is the architecture behind virtually all modern LLMs (GPT, Claude, Gemini, Llama), most of which use the decoder-only variant.

## Detailed Explanation
**The pipeline (decoder-only, as used by GPT-style LLMs)**
```
Token IDs
   │
Embedding lookup (+ position info: RoPE in most modern models)
   │
┌──────── Transformer block ×N ────────┐
│  x = x + MaskedSelfAttention(Norm(x))│  ← tokens exchange information
│  x = x + FeedForward(Norm(x))        │  ← each token processed independently
└──────────────────────────────────────┘
   │
Final Norm → Linear to vocabulary → softmax → P(next token)
```

**Core components**
| Component | Role |
|---|---|
| Token embeddings | turn token IDs into vectors |
| Positional information | tell the model token order (sinusoidal in the original paper; RoPE today) |
| Multi-head self-attention | each token computes weighted mixes of other tokens' information (via Q, K, V) |
| Feed-forward network (MLP) | per-token non-linear processing. Holds much of the model's "knowledge". GELU/SwiGLU |
| Residual connections | `x + sublayer(x)`, giving a gradient highway that allows very deep stacks |
| Layer normalisation | stabilises training. Modern models use pre-norm and often RMSNorm |
| Output head | projects to vocabulary logits |

**Three variants**
| Variant | Attention | Examples | Typical use |
|---|---|---|---|
| Encoder-only | bidirectional | BERT, embedding models | classification, embeddings for retrieval |
| Encoder–decoder | encoder bidirectional + decoder with cross-attention | original Transformer, T5 | translation, summarisation |
| **Decoder-only** | causal (masked) | GPT, Claude, Llama, Gemini, Qwen | general text generation. This is the dominant LLM design |

**Modern refinements (still "Transformers"):** RoPE positions, grouped-query attention (GQA) to shrink the KV cache, SwiGLU feed-forward layers, RMSNorm, FlashAttention kernels, and Mixture-of-Experts (MoE) feed-forward layers that activate only a subset of parameters per token.

## Example
**Analogy:** A meeting where everyone reads the whole document at once. In each round (layer), every participant (token) asks a question (query), checks who has relevant information (keys), collects it (values), then thinks privately (feed-forward). After many rounds, each person has a rich, context-aware understanding. An RNN would instead be a single reader passing notes down a line.

A minimal single block in numpy (one head, no normalisation, for intuition):
```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

rng = np.random.default_rng(0)
n, d = 4, 8                                   # 4 tokens, model width 8
x = rng.normal(size=(n, d))                   # embeddings (+ positions)
Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) * 0.3 for _ in range(4))
W1, W2 = rng.normal(size=(d, 4 * d)) * 0.3, rng.normal(size=(4 * d, d)) * 0.3

Q, K, V = x @ Wq, x @ Wk, x @ Wv
mask = np.triu(np.ones((n, n), bool), k=1)    # causal: no peeking ahead
A = softmax(np.where(mask, -np.inf, Q @ K.T / np.sqrt(d)))
x = x + (A @ V) @ Wo                          # attention + residual
x = x + np.maximum(0, x @ W1) @ W2            # feed-forward + residual
print(x.shape)                                # (4, 8)  same shape: blocks can be stacked
```

## Common Mistakes / Misconceptions
- **"Transformer = attention only."** The feed-forward layers hold roughly two-thirds of the parameters in a standard dense model, and residuals and normalisation are essential.
- **"All LLMs are encoder–decoder like the 2017 paper."** Today's chat LLMs are overwhelmingly decoder-only.
- **"Transformers understand order natively."** Attention alone is order-agnostic. Positional information must be added.
- **"Transformers are only for text."** They power vision (ViT), audio, protein and multimodal models too.

> **Note on overlap:** [Week 12 Q2 – Explain Transformer architecture](../Week-12-Final-Project-and-Revision/02-Explain-Transformer-architecture.md) is the revision-level, end-to-end walkthrough. This answer is the first definition. See also Q3 (*why* Transformers mattered historically) and Q15 (*why* they suit language).

## Key Takeaways
- A Transformer is a stack of blocks: self-attention + feed-forward, each wrapped in residual + norm.
- Attention replaces recurrence, so all positions are processed in parallel and any two tokens can interact directly.
- There are three variants. Decoder-only dominates modern LLMs.
- Modern models add RoPE, GQA, SwiGLU, RMSNorm and sometimes MoE, but the core is the 2017 design.

## Related Questions
- [What problems do traditional RNNs have?](../Week-04-Transformer-Architecture/01-What-problems-do-traditional-RNNs-have.md)
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
- [Explain Transformer architecture.](../Week-12-Final-Project-and-Revision/02-Explain-Transformer-architecture.md)
