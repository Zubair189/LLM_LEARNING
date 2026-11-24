# What is an encoder?

> Week 04 – Transformer Architecture · Question 11

## Short Answer
In a Transformer, an encoder is a stack of layers that reads the whole input sequence at once and turns each token into a contextual representation, using **bidirectional** self-attention (every token can see every other token, left and right) followed by a feed-forward network. Its output is a set of vectors that "understand" the input. They can be used directly (classification, embeddings for search) or passed to a decoder via cross-attention (translation). BERT is the classic encoder-only model.

## Detailed Explanation
**Encoder block** (repeated N times)
```
x ─► Self-Attention (bidirectional, no mask) ─► Add & Norm ─► Feed-Forward ─► Add & Norm ─► out
```
- **No causal mask:** token $i$ attends to all positions $1..n$.
- Input and output have the same shape `(n, d_model)`, so blocks stack.
- Output: one contextual vector per input token.

**How encoder outputs are used**
| Use | How |
|---|---|
| Sequence classification (sentiment, spam) | pool the vectors (or use a special `[CLS]` token) → classifier |
| Token classification (NER) | classifier on each token's vector |
| Sentence embeddings for search/RAG | mean-pool into one vector (e.g. sentence-transformers, most embedding models) |
| Rerankers | encode (query, document) together → relevance score (cross-encoder) |
| Encoder–decoder | decoder cross-attends to the encoder outputs (T5, original Transformer, Whisper's audio encoder) |

**How encoders are trained:** not by next-token prediction (seeing both sides would make that trivial), but by **masked language modelling**: hide ~15% of tokens and predict them from both sides ("The [MASK] sat on the mat" → "cat").

**Encoder vs. decoder at a glance**
| | Encoder | Decoder (GPT-style) |
|---|---|---|
| Attention | bidirectional | causal (masked) |
| Training objective | masked-token prediction | next-token prediction |
| Strength | understanding / representing text | generating text |
| Examples | BERT, RoBERTa, DeBERTa, ModernBERT, embedding models | GPT, Claude, Llama |

**In 2026:** chat LLMs are decoder-only, but encoder models remain the workhorses for embeddings, rerankers and cheap, fast classifiers in production RAG systems. "Encoder" is also used more broadly for any module that turns an input into representations, e.g. vision encoders in multimodal LLMs.

## Example
**Analogy:** A reader who reads the entire paragraph before writing margin notes on each word. Because they've seen the whole thing, each note reflects full context. A decoder is a reader who must annotate each word before seeing what comes next.

**Example:** in "I went to the **bank** to deposit money", the encoder's vector for "bank" uses "deposit money", which comes *after* it. A causal decoder could not use that at the position of "bank".

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

rng = np.random.default_rng(0)
n, d = 5, 8
X = rng.normal(size=(n, d))
Wq, Wk, Wv = (rng.normal(size=(d, d)) * 0.3 for _ in range(3))

A = softmax((X @ Wq) @ (X @ Wk).T / np.sqrt(d))   # NO mask: bidirectional
H = X + A @ (X @ Wv)                              # contextual token vectors (+ residual)

print((A > 0).all())          # True: every token attends to every token, incl. future ones
sentence_vec = H.mean(axis=0) # mean pooling -> one embedding for search/classification
print(sentence_vec.shape)     # (8,)
```

## Common Mistakes / Misconceptions
- **"Encoders generate text."** Encoder-only models produce representations. Generation needs a decoder.
- **"ChatGPT has an encoder."** GPT-style models are decoder-only. Their "encoding" of the prompt happens in the same causal decoder stack.
- **"Encoders are obsolete."** They are still state of the art for many embedding, reranking and classification tasks, at a fraction of LLM cost.
- **Confusing "encoder" with a tokenizer or an embedding table.** The encoder is the full stack of attention + FFN layers.

## Key Takeaways
- An encoder turns an input sequence into contextual vectors using bidirectional self-attention.
- It is trained with masked-language modelling, not next-token prediction.
- It powers classification, embeddings, rerankers, and the input side of encoder–decoder models.
- Decoder-only models dominate generation, while encoders dominate representation tasks.

## Related Questions
- [What is a decoder?](../Week-04-Transformer-Architecture/12-What-is-a-decoder.md)
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [What is the difference between word and sentence embeddings?](../Week-07-Embeddings-and-Vector-Databases/05-What-is-the-difference-between-word-and-sentence-embeddings.md)
