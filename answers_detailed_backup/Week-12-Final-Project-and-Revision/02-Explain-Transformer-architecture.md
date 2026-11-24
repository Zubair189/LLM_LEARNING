# Explain Transformer architecture.

> Week 12 – Final Project and Revision · Question 2

## Short Answer
A Transformer is a neural network that processes all tokens of a sequence in parallel and uses **self-attention** to let every token look at every other relevant token. Each layer combines **multi-head attention** with a **feed-forward network**, wrapped in residual connections and layer normalization, and stacks many such layers. GPT-style LLMs use a **decoder-only** Transformer with **masked (causal) self-attention**, so each token can only attend to earlier tokens, which is exactly what next-token prediction needs.

## Detailed Explanation

### End-to-end flow (decoder-only, GPT-style)

```
 "The cat sat"
      │ tokenize
      ▼
 [464, 3797, 3332]                       token IDs
      │ token embedding  +  positional encoding
      ▼
 ┌────────────────────────────────────────────┐
 │  Transformer block  (× N layers, e.g. 32+) │
 │  ┌──────────────────────────────────────┐  │
 │  │ Masked multi-head self-attention     │  │  Q·Kᵀ/√d → softmax → ·V
 │  └───────────────┬──────────────────────┘  │
 │        + residual, LayerNorm               │
 │  ┌───────────────▼──────────────────────┐  │
 │  │ Feed-forward network (MLP)           │  │
 │  └───────────────┬──────────────────────┘  │
 │        + residual, LayerNorm               │
 └──────────────────┼─────────────────────────┘
                    ▼
       Linear → softmax over vocabulary
                    ▼
       P(next token) → sample → "on"
```

### Components

| Component | Role | Week |
|---|---|---|
| Token embeddings | Map each token ID to a dense vector | 4, 7 |
| Positional encodings | Inject word order (attention alone is order-agnostic) | 4 |
| Query / Key / Value | Each token projects to Q ("what I look for"), K ("what I offer"), V ("my content") | 4 |
| Attention score | `softmax(Q·Kᵀ / √d)` — how much each token attends to others | 3 (dot product, softmax), 4 |
| Multi-head attention | Several attention "heads" in parallel learn different relations (syntax, coreference…) | 4 |
| Causal mask | Blocks attention to future tokens during generation | 4 |
| Feed-forward network | Per-token MLP that transforms features (much of the "knowledge" lives here) | 3 |
| Residual + LayerNorm | Stabilize training of deep stacks | 3 |
| Output head | Projects to vocabulary logits → softmax → probabilities | 3 |

### Encoder vs decoder variants
- **Encoder-only** (BERT): bidirectional attention, good for classification and embeddings.
- **Decoder-only** (GPT, Claude-style chat models): causal attention, good for generation.
- **Encoder-decoder** (original 2017 paper, T5): encoder reads input, decoder generates output with cross-attention; good for translation.

### Why Transformers won over RNNs
- **Parallelism**: all tokens processed at once in training → scales on GPUs.
- **Long-range dependencies**: any token reaches any other in one attention step (RNNs suffer vanishing gradients).
- **Scaling**: performance kept improving with more data, parameters, and compute.

The trade-off: attention cost grows roughly with the **square of sequence length**, which is why context windows are finite and why context management matters in production.

## Example
**Analogy:** A meeting where everyone reads the whole agenda at once. For each word, a participant asks a question (Query), checks everyone's name tags (Keys), and takes notes weighted by relevance (Values). Multiple heads are like several note-takers each focusing on something different (grammar, who-did-what, tone). In GPT-style meetings, you can only listen to people who spoke before you (causal mask).

**Python – scaled dot-product attention with a causal mask (NumPy):**
```python
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

np.random.seed(0)
T, d = 3, 4                        # 3 tokens, dimension 4
X = np.random.randn(T, d)          # embeddings + positions
Wq, Wk, Wv = (np.random.randn(d, d) for _ in range(3))
Q, K, V = X @ Wq, X @ Wk, X @ Wv

scores = Q @ K.T / np.sqrt(d)
mask = np.triu(np.ones((T, T)), k=1).astype(bool)   # block future tokens
scores[mask] = -np.inf
weights = softmax(scores)
out = weights @ V

print(np.round(weights, 2))
# Row 0 attends only to itself; row 2 attends to tokens 0..2, e.g.
# [[1.   0.   0.  ]
#  [0.xx 0.xx 0.  ]
#  [0.xx 0.xx 0.xx]]   (exact values depend on the random seed)
print(out.shape)  # (3, 4)
```

## Common Mistakes / Misconceptions
- **"Attention = memory of past conversations."** Attention only works over tokens currently in the context window.
- **Forgetting positional encodings.** Without them, "dog bites man" and "man bites dog" look the same to attention.
- **"Transformers read left-to-right like RNNs."** Training is parallel; only *generation* is sequential, one token at a time.
- **Assuming all LLMs are encoder-decoder.** Most chat LLMs are decoder-only.

## Key Takeaways
- Transformer = embeddings + positions → N × (attention + feed-forward, with residuals/norm) → vocabulary softmax.
- Self-attention computes `softmax(QKᵀ/√d)V`; multi-head runs several in parallel.
- Decoder-only with causal masking powers GPT-style generation.
- Parallelism and long-range modeling made Transformers scale; quadratic attention cost limits context length.

## Related Questions
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is multi-head attention?](../Week-04-Transformer-Architecture/08-What-is-multi-head-attention.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
- [What problems do traditional RNNs have?](../Week-04-Transformer-Architecture/01-What-problems-do-traditional-RNNs-have.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
- [What is a dot product?](../Week-03-Mathematics-and-Neural-Network-Foundations/05-What-is-a-dot-product.md)
