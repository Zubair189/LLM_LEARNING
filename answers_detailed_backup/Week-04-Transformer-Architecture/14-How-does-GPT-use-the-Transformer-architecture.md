# How does GPT use the Transformer architecture?

> Week 04 – Transformer Architecture · Question 14

## Short Answer
GPT (Generative Pre-trained Transformer) uses only the **decoder** side of the Transformer: a deep stack of blocks, each with **causal (masked) self-attention** and a feed-forward network, wrapped in residual connections and layer normalisation. It is **pre-trained** to predict the next token on massive amounts of text, then **post-trained** (instruction tuning, RLHF and similar) to follow instructions. At inference it generates **autoregressively**: predict a token, append it, repeat, using a KV cache for speed.

## Detailed Explanation
**Architecture (decoder-only)**
```
token IDs ─► token embedding (+ position: learned in GPT-2/3; RoPE in most modern models)
          ─► [ x + CausalMHA(LN(x)) ; x + MLP(LN(x)) ] × N layers
          ─► final LN ─► linear to vocab (often tied to embeddings) ─► softmax
          ─► P(next token | all previous tokens)
```
What GPT removes from the original Transformer: the encoder and cross-attention. What it keeps: masked self-attention, FFN, residuals, normalisation.

**Training stages**
| Stage | Data | Objective |
|---|---|---|
| Pre-training | trillions of tokens of web, books, code | next-token cross-entropy, $-\sum_t \log P(x_t\mid x_{<t})$ |
| Supervised fine-tuning (SFT) | instruction → response examples | same loss, on responses |
| Preference / RL tuning | human or AI preferences, verifiable rewards | RLHF, DPO, RL for reasoning |

**Inference loop**
1. Tokenise the prompt and run one forward pass over all prompt tokens ("prefill"), filling the KV cache.
2. Take the logits at the last position, apply temperature / top-p, and choose a token.
3. Append it, run a forward pass on just that token (reusing the KV cache), and repeat until a stop token or max length ("decode" phase).

**Evolution of the GPT family (illustrative)**
| Model | Year | Notable |
|---|---|---|
| GPT-1 | 2018 | 117M params; showed generative pre-training + fine-tuning works |
| GPT-2 | 2019 | 1.5B; strong zero-shot text generation; pre-norm |
| GPT-3 | 2020 | 175B; few-shot in-context learning |
| ChatGPT / GPT-3.5 | 2022 | instruction tuning + RLHF → conversational assistant |
| GPT-4 and later | 2023+ | multimodal, longer contexts, reasoning; architecture details not fully disclosed |

Other decoder-only LLMs (Claude, Llama, Gemini, Qwen, Mistral, DeepSeek) follow the same recipe, with variations such as RoPE, GQA, SwiGLU, RMSNorm and Mixture-of-Experts.

## Example
**Analogy:** An extremely well-read autocomplete. Having read a large portion of the internet, it has learned which word is likely next in almost any context. Post-training then teaches it to use that skill as a helpful assistant rather than just continuing text.

A toy GPT forward pass + greedy generation loop (random, *untrained* weights, so the output is nonsense; the point is the mechanics):
```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

rng = np.random.default_rng(0)
vocab = ["<bos>", "the", "cat", "sat", "on", "mat"]
V, d = len(vocab), 8
E = rng.normal(size=(V, d)) * 0.5          # token embeddings (tied as output head)
P = rng.normal(size=(16, d)) * 0.1         # learned positional embeddings (GPT-2 style)
Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) * 0.3 for _ in range(4))

def gpt_forward(ids):
    n = len(ids)
    x = E[ids] + P[:n]
    Q, K, Vv = x @ Wq, x @ Wk, x @ Wv
    S = Q @ K.T / np.sqrt(d)
    S = np.where(np.triu(np.ones((n, n), bool), 1), -np.inf, S)  # causal mask
    x = x + softmax(S) @ Vv @ Wo                                 # one attention block
    return x @ E.T                                               # logits for every position

ids = [0, 1, 2]                            # "<bos> the cat"
for _ in range(3):                         # autoregressive loop
    logits = gpt_forward(ids)
    ids.append(int(np.argmax(logits[-1]))) # greedy: take the most likely next token
print(logits.shape)                        # (5, 6)  one distribution per position
print([vocab[i] for i in ids])             # ['<bos>', 'the', 'cat', 'cat', 'cat', 'cat']  untrained -> repetitive
```
Training would adjust all these matrices so that `logits[t]` assigns high probability to `ids[t+1]`.

## Common Mistakes / Misconceptions
- **"GPT uses the full encoder–decoder Transformer."** It is decoder-only.
- **"GPT looks up answers in a database."** It generates from patterns stored in its weights (unless connected to tools or retrieval).
- **"GPT produces the whole answer in one pass."** One forward pass yields one token.
- **"Pre-training alone makes a chatbot."** Instruction tuning and preference training are what make it follow instructions.

> **Note on overlap:** [Week 2 Q4 – What does GPT stand for?](../Week-02-Introduction-to-LLMs-and-Generative-AI/04-What-does-GPT-stand-for.md) covers the name. This question covers the architecture behind it.

## Key Takeaways
- GPT = a decoder-only Transformer with causal self-attention.
- It is pre-trained on next-token prediction, then instruction- and preference-tuned.
- It generates autoregressively: prefill the prompt, then decode token by token with a KV cache.
- The same blueprint underlies nearly all modern chat LLMs.

## Related Questions
- [What is a decoder?](../Week-04-Transformer-Architecture/12-What-is-a-decoder.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [What does GPT stand for?](../Week-02-Introduction-to-LLMs-and-Generative-AI/04-What-does-GPT-stand-for.md)
- [What is the KV cache and why does it speed up generation? (bonus)](../Week-04-Transformer-Architecture/16-BONUS-What-is-the-KV-cache-and-why-does-it-speed-up-generation.md)
