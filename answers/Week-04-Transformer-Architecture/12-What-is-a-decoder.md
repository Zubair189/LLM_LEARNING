# What is a decoder?

> Week 04 – Transformer Architecture · Question 12

## In One Line
A decoder is the part of a Transformer that **writes text one token at a time**, where each token can only look at the tokens **before** it.

## Simple Explanation
A decoder is a stack of blocks, like an encoder, but with one key rule: **no peeking ahead**. Each position may only use earlier positions. This is done with **masked self-attention** (next question).

**How it generates text (called autoregressive generation):**
1. Start with the prompt, e.g. "The cat".
2. Predict the next token: "sat".
3. Add it to the input: "The cat sat".
4. Predict again: "on". Repeat until an end token.

**How it is trained:** we feed in a real sentence. Because of the no-peeking rule, **every position** can practise "guess the next token" at the same time, in one pass. So training is fast and parallel, even though generation is step by step. (This trick of feeding the true sentence is called **teacher forcing**.)

**Two versions:**
- **In the original Transformer (encoder–decoder):** the decoder also has **cross-attention**, which lets it look at the encoder's output, e.g. the source sentence when translating.
- **Decoder-only (GPT, Claude, Llama):** no encoder and no cross-attention. The prompt and the answer are just one long stream of text.

**Why decoder-only won for chatbots:** one simple goal (predict the next token) works on any text, so it scales easily, and any task can be written as "continue this text".

**Link to LLMs:** every modern chat LLM is a decoder-only Transformer.

## Real-Life Example
Writing a story one word at a time with a sheet of paper covering everything you haven't written yet. You can reread all your earlier words, but you can't peek ahead.

## Code Example
How one training sentence becomes several "guess the next token" exercises:

```python
tokens = ["The", "cat", "sat", "down"]

for i in range(1, len(tokens)):
    seen = tokens[:i]               # what the model is allowed to see
    target = tokens[i]              # what it must predict
    print(seen, "->", target)
# ['The'] -> cat
# ['The', 'cat'] -> sat
# ['The', 'cat', 'sat'] -> down
```

**What happened:** one 4-token sentence gave 3 training exercises. Each one sees only the earlier tokens. A decoder does all of these in a single parallel pass.

## Watch Out
- A decoder does **not** need an encoder. GPT-style models have none.
- Training is parallel; only **generation** is one token at a time.

## Remember
- Decoder = generates text, one token at a time.
- Each token sees only earlier tokens (causal mask).
- Modern LLMs are decoder-only.

## Related Questions
- [What is an encoder?](../Week-04-Transformer-Architecture/11-What-is-an-encoder.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
- [What is the KV cache and why does it speed up generation? (bonus)](../Week-04-Transformer-Architecture/16-BONUS-What-is-the-KV-cache-and-why-does-it-speed-up-generation.md)
