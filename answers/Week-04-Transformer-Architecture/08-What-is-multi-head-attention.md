# What is multi-head attention?

> Week 04 – Transformer Architecture · Question 8

## In One Line
Multi-head attention runs **several attentions side by side**, so each "head" can look for a **different kind of relationship** between words at the same time.

## Simple Explanation
One attention gives each word **one** set of weights over the other words. But words relate in many ways at once: grammar ("which noun is the subject of this verb?"), meaning ("what does 'it' refer to?"), position ("what was the previous word?"). One set of weights has to squash all of that together.

**Multi-head attention** fixes this:
1. Split the work into several **heads** (for example 8 or 32).
2. Each head has its **own** learned query, key and value weights, working on a **smaller slice** of the numbers.
3. Each head does its own attention and finds its own pattern.
4. The heads' outputs are **joined back together** (concatenated) and mixed by one more weight matrix, called the **output projection** (`W_O`).

Because each head works on a smaller slice, the total cost is about the **same** as one big attention. You get several viewpoints for roughly the same price.

Researchers have seen heads that specialise, e.g. one tracks the previous word and another copies patterns seen earlier in the text. But not every head learns something clearly meaningful.

**Link to LLMs:** every attention layer in an LLM is multi-head. Many modern LLMs let several query heads **share** the same keys and values (called **grouped-query attention**, GQA) to save memory during generation.

## Real-Life Example
A panel of experts reads the same document: a grammar checker, a fact-checker and a style editor. Each highlights different connections. Then their notes are combined into one report. The experts are the heads; combining the notes is `W_O`.

## Code Example
Two heads, each looking at a different half of the word vectors. (In a real model each head gets its own learned weights; here we just split the numbers to keep it simple.)

```python
import math

def attention_weights(q, keys):              # dot product + softmax
    exps = [math.exp(sum(a * b for a, b in zip(q, k))) for k in keys]
    return [round(e / sum(exps), 2) for e in exps]

# 3 words, 4 numbers each: first 2 = "grammar" features, last 2 = "meaning" features
X = [[1, 0, 0, 2],     # word 0
     [1, 0, 0, 0],     # word 1: same grammar as word 0
     [0, 1, 0, 2]]     # word 2: similar meaning to word 0

head1 = [x[:2] for x in X]                   # head 1 sees only the first half
head2 = [x[2:] for x in X]                   # head 2 sees only the second half

print(attention_weights(head1[0], head1))    # [0.42, 0.42, 0.16]
print(attention_weights(head2[0], head2))    # [0.5, 0.01, 0.5]
```

**What happened:** for word 0, head 1 focused on word 1 (same grammar) and head 2 focused on word 2 (similar meaning). Two heads, two different relationships, at once.

## Watch Out
- More heads does **not** mean more parameters; the same width is just split into more, smaller heads.
- Don't forget `W_O`: it is what lets the heads' findings mix together.

## Remember
- Several attentions in parallel, each on a smaller slice.
- Different heads can catch different relationships.
- Outputs are joined and mixed by `W_O`.

## Related Questions
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is the KV cache and why does it speed up generation? (bonus)](../Week-04-Transformer-Architecture/16-BONUS-What-is-the-KV-cache-and-why-does-it-speed-up-generation.md)
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
