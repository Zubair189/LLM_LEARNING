# What are positional encodings?

> Week 04 – Transformer Architecture · Question 9

## Short Answer
Positional encodings give a Transformer information about the order of tokens. Self-attention by itself treats its input as an unordered set: shuffle the tokens and each one's output is unchanged apart from being shuffled. So position must be injected explicitly. The original Transformer added fixed sinusoidal vectors to the token embeddings. Most modern LLMs instead use **RoPE** (Rotary Position Embedding), which rotates query and key vectors so that attention scores depend on the *relative* distance between tokens.

## Detailed Explanation
**Why needed:** "dog bites man" and "man bites dog" contain the same tokens. Without positional information, attention would produce the same set of representations for both.

**Main approaches**
| Method | How | Used by | Notes |
|---|---|---|---|
| Sinusoidal (absolute, fixed) | add $PE_{pos}$ to the embedding | original Transformer | no parameters, smooth |
| Learned absolute | a trainable vector per position, added | BERT, GPT-2 | can't go beyond the trained max length |
| **RoPE** (relative, rotary) | rotate Q and K by a position-dependent angle | Llama, Qwen, Mistral, most modern LLMs | relative by construction; context extension via scaling (NTK, YaRN) |
| ALiBi | add a linear distance penalty to attention scores | BLOOM, MPT | good length extrapolation |

**Sinusoidal formula**
$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$
Each pair of dimensions is a sine/cosine "clock" at a different frequency: fast-ticking clocks distinguish nearby positions, slow ones distinguish distant ones, like the second, minute and hour hands of a clock.

**RoPE in one idea:** treat each pair of dimensions of $q$ and $k$ as a 2-D point and rotate it by angle $pos\cdot\theta_i$. Since rotating $q$ by $m\theta$ and $k$ by $n\theta$ gives a dot product depending only on $(m-n)\theta$,
$$\langle R_m q,\, R_n k\rangle = \langle q,\, R_{n-m} k\rangle,$$
attention scores naturally encode **relative** position. RoPE is applied inside every attention layer (to Q and K only), not added once at the input.

## Example
**Analogy:** Seats in a theatre. Two people (tokens) may be identical, but their seat numbers (positions) tell you who sits next to whom. RoPE is like giving each person a compass needle rotated by their seat number: comparing two needles tells you how many seats apart they are, wherever they sit.

```python
import numpy as np

def sinusoidal_pe(n_pos, d):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(0, d, 2)[None, :]
    angle = pos / (10000 ** (i / d))
    pe = np.zeros((n_pos, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe

print(sinusoidal_pe(4, 4).round(3))
# [[ 0.     1.     0.     1.   ]    position 0
#  [ 0.841  0.54   0.01   1.   ]    position 1: fast clock moved a lot, slow clock barely
#  [ 0.909 -0.416  0.02   1.   ]
#  [ 0.141 -0.99   0.03   1.   ]]

def rope(x, pos, base=10000):
    d = x.shape[-1]
    theta = pos / base ** (np.arange(0, d, 2) / d)
    c, s = np.cos(theta), np.sin(theta)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * c - x[1::2] * s
    out[1::2] = x[0::2] * s + x[1::2] * c
    return out

q, k = np.array([1.0, 0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.5, 0.5])
print(round(rope(q, 5) @ rope(k, 3), 4), round(rope(q, 12) @ rope(k, 10), 4))
# 0.0938 0.0938   same score: only the distance (2) matters, not absolute position
```

## Common Mistakes / Misconceptions
- **"Transformers know word order automatically."** They don't. Remove positional information and order is lost.
- **"Positional encodings are always added to the embeddings."** That's true for sinusoidal and learned absolute encodings, but RoPE rotates Q and K inside attention, and ALiBi modifies the scores.
- **"A model works at any context length."** Models degrade beyond their trained length unless position scaling (e.g. YaRN) or long-context training is applied.
- **Applying RoPE to values.** It is applied to queries and keys only.

## Key Takeaways
- Self-attention is order-blind, so positional information is required.
- Sinusoidal encodings add multi-frequency sine/cosine vectors to the input embeddings.
- RoPE, the modern default, rotates Q and K so attention depends on relative distance.
- The choice of position scheme strongly affects long-context behaviour and extension.

## Related Questions
- [What are token embeddings?](../Week-04-Transformer-Architecture/10-What-are-token-embeddings.md)
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
