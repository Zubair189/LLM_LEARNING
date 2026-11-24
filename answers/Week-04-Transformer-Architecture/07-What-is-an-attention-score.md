# What is an attention score?

> Week 04 – Transformer Architecture · Question 7

## In One Line
An attention score is **a number that says how relevant one word is to another**; softmax then turns these scores into weights that add up to 1.

## Simple Explanation
Each word has a **query** (what it's looking for), and every word has a **key** (its label). The attention score is how well a query matches a key.

The formula, shown once:

```
score = (query · key) / √d
```

- **query · key:** the dot product: multiply matching numbers and add them up. More similar vectors → bigger score.
- **d:** how many numbers are in each vector. Dividing by √d keeps the scores from getting too large. Very large scores would make softmax put almost all weight on one word, which makes learning hard.

**Two words to keep apart:**
- **Score:** the raw number before softmax. It can be any value, even negative.
- **Weight:** the number after softmax. Between 0 and 1, and all weights for one word add up to 1.

**Worked by hand** (query `[1, 2]`, three keys, d = 2):
1. Dot products: `[1,0]` → 1, `[1,1]` → 3, `[0,2]` → 4
2. Divide by √2 ≈ 1.41: **[0.71, 2.12, 2.83]**
3. Softmax: **[0.07, 0.31, 0.62]**

So this word pays most attention (62%) to the third word.

**Masks** work on the scores **before** softmax. For example, GPT-style models set the scores of future words to minus infinity, so their weights become exactly 0.

**Link to LLMs:** every attention layer computes a score for every pair of words, then turns them into weights.

## Real-Life Example
A recruiter gives each CV a raw rating (the scores). Then she turns the ratings into "what percentage of my interview time goes to each candidate" (the weights). Better ratings get a bigger share, and the shares add up to 100%.

## Code Example
```python
import math

q = [1, 2]                                    # query of one word
keys = [[1, 0], [1, 1], [0, 2]]               # keys of three words
d = len(q)                                    # vector size = 2

raw = [sum(a * b for a, b in zip(q, k)) for k in keys]   # dot products
scaled = [s / math.sqrt(d) for s in raw]                  # divide by sqrt(d)
exps = [math.exp(s) for s in scaled]
weights = [e / sum(exps) for e in exps]                   # softmax

print(raw)                                    # [1, 3, 4]
print([round(s, 2) for s in scaled])          # [0.71, 2.12, 2.83]
print([round(w, 2) for w in weights])         # [0.07, 0.31, 0.62]
```

**What happened:** raw scores were scaled, then turned into weights. The key most aligned with the query got the biggest weight.

## Watch Out
- Don't mix up **scores** (before softmax) and **weights** (after softmax).
- Masks must be applied to scores **before** softmax, otherwise the weights won't add up to 1.

## Remember
- Score = query · key / √d.
- Softmax turns scores into weights that sum to 1.
- Bigger score → more of that word's value is used.

## Related Questions
- [What is a dot product?](../Week-03-Mathematics-and-Neural-Network-Foundations/05-What-is-a-dot-product.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
