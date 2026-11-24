# What are token embeddings?

> Week 04 – Transformer Architecture · Question 10

## Short Answer
Token embeddings are learned vectors that represent each token in a model's vocabulary. The model holds an embedding matrix of shape `(vocab_size, d_model)`, and converting a token ID into its vector is simply a row lookup. These vectors are trained along with the rest of the network, so tokens used in similar contexts end up with similar vectors. They are the first layer of every Transformer: they turn discrete symbols into numbers that attention and MLPs can process.

## Detailed Explanation
**Pipeline**
```
"The cat sat"  →  tokenizer  →  [464, 3797, 3332]  →  embedding lookup  →  3 × d_model matrix
                                                       (+ position info)  →  Transformer blocks
```

**The embedding matrix**
- Shape `(V, d)`, e.g. vocab 128k × 4096 is about 0.5B parameters, a significant share of a small model.
- Lookup `E[token_id]` is mathematically the same as multiplying a one-hot vector by $E$, but much cheaper.
- Starts random. Training adjusts it by backprop like any other weight.

**Why not use the token ID or a one-hot vector directly?**
| Representation | Problem |
|---|---|
| Integer ID (e.g. 3797) | implies a false ordering and arithmetic ("cat" − 1 is meaningless) |
| One-hot (length V) | huge, sparse, and every pair of tokens is equally distant, so there is no notion of similarity |
| **Dense embedding** | compact, and similar tokens are close together; geometry encodes meaning |

**Static vs. contextual**
- The **input token embedding** is static: "bank" always starts as the same vector.
- After attention layers, the **hidden states** become contextual: "bank" in "river bank" and "bank loan" diverge.

**Weight tying:** many models reuse the embedding matrix (transposed) as the output projection to vocabulary logits. This saves parameters and links input and output token representations. Many large modern models untie them.

**Related embeddings**
- *Positional* information is added to or combined with token embeddings (sinusoidal, learned, or RoPE inside attention).
- *Sentence or document embeddings* used for search (Week 7) are produced by separate embedding models. They pool hidden states into one vector per text. Don't confuse them with the per-token input embedding table.

## Example
**Analogy:** A library catalogue where every word has a coordinate on a giant map. "Cat" and "dog" live in the same neighbourhood (pets), and "sat" lives elsewhere (verbs). The model learns where to place each word so that its later layers can use the map.

```python
import numpy as np

vocab = {"the": 0, "cat": 1, "sat": 2, "dog": 3}
E = np.array([[0.1, 0.0, 0.2],    # the
              [0.9, 0.8, 0.1],    # cat
              [0.0, 0.3, 0.9],    # sat
              [0.8, 0.9, 0.2]])   # dog      (V=4, d=3; learned in practice)

ids = [vocab[t] for t in ["the", "cat", "sat"]]
print(ids)                 # [0, 1, 2]
print(E[ids])              # lookup = 3 x 3 matrix fed to the Transformer
# [[0.1 0.  0.2]
#  [0.9 0.8 0.1]
#  [0.  0.3 0.9]]

print(np.eye(4)[1] @ E)    # [0.9 0.8 0.1]  one-hot @ E == row lookup

cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(round(cos(E[1], E[3]), 3), round(cos(E[1], E[2]), 3))
# 0.99 0.288   cat~dog similar, cat~sat not
```

## Common Mistakes / Misconceptions
- **"Embeddings are hand-designed or fixed."** They are learned parameters, updated during training.
- **"An embedding captures a word's full meaning."** The input embedding is context-free. Context is added by attention layers.
- **"Token = word."** Tokens are often sub-words, so the embedding of "un" is shared across "undo", "unhappy" and so on.
- **Confusing token embeddings with the embedding-model vectors used in RAG.** Those are a different (though related) concept.

> **Note on overlap:** [Week 7 Q1 – What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md) covers embeddings in general and text embeddings for search. This question covers the per-token lookup table at the input of a Transformer.

## Key Takeaways
- Token embeddings map token IDs to dense, learned vectors via a `(vocab, d_model)` lookup table.
- They replace sparse one-hot vectors with compact representations where geometry reflects meaning.
- They are static at the input and become contextual through the Transformer layers.
- They are often tied with the output projection, and are combined with positional information.

## Related Questions
- [What is a token?](../Week-02-Introduction-to-LLMs-and-Generative-AI/06-What-is-a-token.md)
- [What are positional encodings?](../Week-04-Transformer-Architecture/09-What-are-positional-encodings.md)
- [What is a vector?](../Week-03-Mathematics-and-Neural-Network-Foundations/02-What-is-a-vector.md)
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
