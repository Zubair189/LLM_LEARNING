# What problems do traditional RNNs have?

> Week 04 – Transformer Architecture · Question 1

## Short Answer
Recurrent neural networks (RNNs) process a sequence one token at a time, compressing everything seen so far into a fixed-size hidden state. This causes three main problems. Gradients vanish or explode over long sequences, so long-range dependencies are hard to learn. Computation is sequential, so training cannot be parallelised across time steps on GPUs. And the fixed-size state is an information bottleneck for long inputs. Transformers address all three with attention.

## Detailed Explanation
**How an RNN works**
$$h_t = \tanh(h_{t-1}W_h + x_tW_x + b)$$
```
x1 → [RNN] → h1 → [RNN] → h2 → [RNN] → h3 → ... → hT
             ↑ x2         ↑ x3
```
Each hidden state depends on the previous one, like a chain.

**Problem 1: vanishing and exploding gradients.** Backpropagation through time multiplies by (roughly) the same Jacobian once per step. Over $T$ steps the gradient scales like $w^T$:
- $|w| < 1$: shrinks exponentially, so early tokens get almost no learning signal (vanishing).
- $|w| > 1$: grows exponentially, making training unstable (exploding, usually patched with gradient clipping).

In practice, vanilla RNNs struggle to connect words more than roughly 10–20 steps apart.

**Problem 2: no parallelism over time.** $h_t$ can't be computed until $h_{t-1}$ exists, so a 1,000-token sequence needs 1,000 sequential steps. GPUs, which thrive on large parallel matrix multiplications, sit mostly idle. This made it impractical to train RNNs on internet-scale data.

**Problem 3: fixed-size bottleneck.** In seq2seq translation, the entire source sentence had to fit into one vector $h_T$. Quality dropped sharply for long sentences. The original attention mechanism (Bahdanau et al., 2014) was invented to fix exactly this.

**Partial fixes and their limits**
| Fix | What it helped | What remained |
|---|---|---|
| LSTM / GRU (gates, cell state) | much better gradient flow, hundreds of steps | still sequential, still a bottleneck |
| Gradient clipping | exploding gradients | not vanishing ones |
| Attention on top of an RNN | bottleneck | still sequential |

**The Transformer's answer (2017):** drop recurrence entirely. Self-attention connects every token to every other token directly (path length 1), and all positions are processed in parallel during training.

**2026 footnote:** recurrent ideas returned in *linear-time* models (state-space models like Mamba, RWKV, and hybrid attention/SSM architectures). These are designed to train in parallel while keeping constant-memory inference, which is interesting for very long contexts. Pure classic RNNs are not used for modern LLMs.

## Example
**Analogy:** The game of "telephone". A message is whispered person to person down a long line. By the end, details from the start are lost (vanishing information), and you can't speed it up because each person must wait for the previous one (sequential processing).

**Long-range dependency:** "The **keys** that I left on the kitchen table next to the old newspapers **were** missing." The verb must agree with "keys", 10 words earlier.

```python
import numpy as np

# How a gradient factor w behaves after T steps of backprop through time
for w in [0.5, 0.9, 1.0, 1.1]:
    print(w, [f"{w**T:.2e}" for T in (10, 50, 100)])
# 0.5 ['9.77e-04', '8.88e-16', '7.89e-31']   vanishes
# 0.9 ['3.49e-01', '5.15e-03', '2.66e-05']   vanishes (slower)
# 1.0 ['1.00e+00', '1.00e+00', '1.00e+00']
# 1.1 ['2.59e+00', '1.17e+02', '1.38e+04']   explodes

# A minimal RNN: note the unavoidable sequential loop
rng = np.random.default_rng(0)
d = 3
Wh, Wx = rng.normal(size=(d, d)) * 0.3, rng.normal(size=(d, d)) * 0.3
h = np.zeros(d)
for x in rng.normal(size=(5, d)):     # step t needs h from step t-1
    h = np.tanh(h @ Wh + x @ Wx)
print(h.round(3))                     # [0.562 0.216 0.17 ]
```

## Common Mistakes / Misconceptions
- **"LSTMs solved the problem completely."** They greatly improved gradient flow but remained sequential and still degraded on very long sequences.
- **"RNNs can't handle long sequences at all."** They can process them. The problem is learning dependencies across them.
- **"Transformers have no weaknesses here."** Attention costs $O(n^2)$ in sequence length, which is why efficient-attention methods and SSM hybrids exist.

## Key Takeaways
- RNNs suffer from vanishing and exploding gradients over long sequences.
- Their sequential computation prevents parallel training on GPUs.
- A single fixed-size hidden state bottlenecks information.
- Transformers replaced recurrence with attention: direct token-to-token links and full parallelism.

## Related Questions
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
- [Why did Transformers become important?](../Week-04-Transformer-Architecture/03-Why-did-Transformers-become-important.md)
- [What is attention?](../Week-04-Transformer-Architecture/04-What-is-attention.md)
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
