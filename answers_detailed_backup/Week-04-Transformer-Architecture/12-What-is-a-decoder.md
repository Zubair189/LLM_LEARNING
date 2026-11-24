# What is a decoder?

> Week 04 – Transformer Architecture · Question 12

## Short Answer
In a Transformer, a decoder is a stack of layers that generates output one token at a time, where each position may only attend to earlier positions. This is enforced by **masked (causal) self-attention**. In the original encoder–decoder design, each decoder layer also has a **cross-attention** sub-layer that looks at the encoder's output. Modern LLMs such as GPT, Claude and Llama are **decoder-only**: they drop the encoder and cross-attention, and treat the prompt and the response as one continuous sequence.

## Detailed Explanation
**Original decoder block** (encoder–decoder, e.g. translation):
```
y ─► Masked Self-Attn ─► Add&Norm ─► Cross-Attn (Q from decoder, K,V from encoder) ─► Add&Norm ─► FFN ─► Add&Norm
```

**Decoder-only block** (GPT-style, today's LLMs; pre-norm):
```
x ─► x + MaskedSelfAttn(Norm(x)) ─► x + FFN(Norm(x))
```

**Key properties**
| Property | Explanation |
|---|---|
| Causal mask | position $t$ sees tokens $1..t$, never the future |
| Autoregressive | generate token, append it to the input, repeat |
| Training objective | next-token prediction (cross-entropy) on every position at once |
| Parallel training / sequential generation | training uses teacher forcing (all positions in one pass), while inference produces one token per forward pass |

**Training vs. inference**
- *Training:* feed the true sequence. The mask lets every position learn "predict the next token" simultaneously. One forward pass gives $n$ predictions.
- *Inference:* start from the prompt, predict one token, append it, predict the next, and so on. The **KV cache** stores past keys and values so each step only processes the new token.

**Why decoder-only won for LLMs**
- One simple objective (next token) works on any raw text, so it scales with data.
- Prompt and answer share one stream, so any task can be phrased as "continue this text". That enables in-context learning and chat.
- Simpler architecture to scale, and efficient serving with KV caching.

**Encoder–decoder still appears** where there is a clearly separate input modality or sequence: speech recognition (Whisper), some translation and summarisation models (T5 family).

## Example
**Analogy:** Writing a story one word at a time with a sheet covering everything you haven't written yet. You can reread all previous words (attention to the past), but you can never peek ahead. In translation, you also glance at the source sentence (cross-attention).

**Generation trace** for prompt "The cat":
```
step 1: input [The, cat]            → predicts "sat"
step 2: input [The, cat, sat]       → predicts "on"
step 3: input [The, cat, sat, on]   → predicts "the"   ... until an end token
```

```python
import numpy as np

n = 4
mask = np.tril(np.ones((n, n), dtype=int))   # 1 = may attend, 0 = blocked
print(mask)
# [[1 0 0 0]     token 1 sees only itself
#  [1 1 0 0]     token 2 sees tokens 1-2
#  [1 1 1 0]
#  [1 1 1 1]]    token 4 sees everything before it

# Teacher forcing: one sequence gives n-1 training pairs in a single pass
tokens = ["The", "cat", "sat", "down"]
print(list(zip(tokens[:-1], tokens[1:])))
# [('The', 'cat'), ('cat', 'sat'), ('sat', 'down')]   (each input sees its full prefix)
```

## Common Mistakes / Misconceptions
- **"A decoder needs an encoder."** Decoder-only models have no encoder and no cross-attention.
- **"Decoders are slow to train because they generate sequentially."** Training is fully parallel. Only inference is sequential.
- **"Decoders only read the prompt left-to-right, so they understand it worse."** Causal attention limits each position, but deep stacks still build strong prompt representations. That trade-off is what buys scalable generation.
- **Forgetting the KV cache** when reasoning about inference speed and memory.

## Key Takeaways
- A decoder generates autoregressively using causal (masked) self-attention.
- The original design adds cross-attention to an encoder. Decoder-only models drop it.
- Training is parallel via teacher forcing. Inference is token-by-token with a KV cache.
- Decoder-only Transformers are the dominant architecture for modern LLMs.

## Related Questions
- [What is an encoder?](../Week-04-Transformer-Architecture/11-What-is-an-encoder.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
- [What is the KV cache and why does it speed up generation? (bonus)](../Week-04-Transformer-Architecture/16-BONUS-What-is-the-KV-cache-and-why-does-it-speed-up-generation.md)
