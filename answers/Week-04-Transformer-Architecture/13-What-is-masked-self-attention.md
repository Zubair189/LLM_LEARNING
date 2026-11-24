# What is masked self-attention?

> Week 04 – Transformer Architecture · Question 13

## In One Line
Masked self-attention is self-attention where **each token can only look at itself and earlier tokens, never at future ones**.

## Simple Explanation
In normal self-attention, every word can see every other word. But a text generator must **not** see the future, because when it writes word 3, word 4 doesn't exist yet. Masking blocks the future.

**Who can see whom (4 tokens):**
```
           sees: t1  t2  t3  t4
token 1:         ✓   ✗   ✗   ✗
token 2:         ✓   ✓   ✗   ✗
token 3:         ✓   ✓   ✓   ✗
token 4:         ✓   ✓   ✓   ✓
```
The allowed part is a triangle, so it is also called a **causal** or **triangular** mask.

**How it's done, step by step:**
1. Compute the attention scores as usual.
2. For every future position, replace the score with **minus infinity**.
3. Apply softmax. Since e^(−∞) = 0, future tokens get a weight of exactly **0**, and the remaining weights still add up to 1.

**Why minus infinity and not 0?** A score of 0 still becomes e⁰ = 1 inside softmax, so the future would still get some weight.

**Why it's needed in training:** we feed the whole real sentence at once. Without the mask, the position predicting word 3 could simply look at word 3, the loss would be low, and the model would learn nothing useful. The mask makes training match real generation.

**Link to LLMs:** this is the defining feature of GPT-style (decoder) models.

## Real-Life Example
An exam where each question is revealed one at a time. You can look back at earlier questions and answers, but the next page is sealed. The mask is the seal.

## Code Example
```python
import math

scores = [[2, 1, 0, 1],
          [1, 2, 1, 0],
          [0, 1, 2, 1],
          [1, 0, 1, 2]]                   # raw scores for 4 tokens

for i, row in enumerate(scores):
    # mask: tokens after position i get minus infinity
    masked = [s if j <= i else -math.inf for j, s in enumerate(row)]
    exps = [math.exp(s) for s in masked]  # exp(-inf) = 0
    print([round(e / sum(exps), 2) for e in exps])
# [1.0, 0.0, 0.0, 0.0]       token 1 sees only itself
# [0.27, 0.73, 0.0, 0.0]
# [0.09, 0.24, 0.67, 0.0]
# [0.2, 0.07, 0.2, 0.53]     token 4 sees all four
```

**What happened:** every future position got weight 0, and each row still sums to 1 (up to rounding).

## Watch Out
- Apply the mask **before** softmax. Zeroing weights afterwards breaks the "sums to 1" rule.
- This is different from BERT's "masked language modelling", which hides input **words**. Here we block attention **links**.

## Remember
- Each token sees only itself and the past.
- Future scores → minus infinity → weight 0.
- It lets decoders train on all positions at once without cheating.

## Related Questions
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [What is a decoder?](../Week-04-Transformer-Architecture/12-What-is-a-decoder.md)
- [What is an attention score?](../Week-04-Transformer-Architecture/07-What-is-an-attention-score.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
