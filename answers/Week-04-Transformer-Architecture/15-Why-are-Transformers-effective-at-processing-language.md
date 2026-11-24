# Why are Transformers effective at processing language?

> Week 04 – Transformer Architecture · Question 15

## In One Line
Language depends heavily on **context and far-apart connections**, and self-attention lets every word **look directly at every other word** to work these out.

## Simple Explanation
Language has some tricky features. Here is how Transformers handle each one:

1. **Words connect over long distances.** In "The **keys** that I left on the table **were** missing", "were" must match "keys", many words earlier. Attention links any two words in one step, however far apart.
2. **Words change meaning with context.** "Bank" means something different in "river bank" and "robbed the bank". Each word's numbers are recalculated from its neighbours, so the same word gets a different meaning in each sentence.
3. **Many relationships at once.** Grammar, who "she" refers to, the topic… **Multiple attention heads** can track different relationships at the same time.
4. **Layers of structure.** Words form phrases, phrases form sentences, sentences form ideas. **Stacking many layers** lets the model build from simple patterns (in lower layers) to meaning (in higher layers).
5. **Word order matters.** **Positional information** tells the model which word came where.
6. **Language is huge and varied.** Because Transformers train in parallel, they can learn from trillions of tokens, enough to pick up a vast range of patterns.

**Limits:** attention gets expensive for very long texts, and models can still state wrong things confidently (hallucinate).

**Link to LLMs:** these strengths are why every major LLM is built on Transformers.

## Real-Life Example
A good reader doesn't read strictly word by word with a fading memory. They glance back and forth, connecting "she" to "Dr. Patel" three sentences earlier, while also keeping track of tone and topic. Each attention head is like one of those glances.

## Code Example
The same word "bank" gets a different meaning depending on its neighbour:

```python
import math

def contextualise(seq):                  # simple self-attention (Q = K = V = the vectors)
    out = []
    for x in seq:
        exps = [math.exp(sum(a * b for a, b in zip(x, y)) / 2) for y in seq]
        w = [e / sum(exps) for e in exps]                    # attention weights
        out.append([round(sum(w[j] * seq[j][k] for j in range(len(seq))), 2) for k in range(4)])
    return out

bank  = [1, 1, 0, 0]      # the same starting vector both times
river = [2, 0, 1, 0]      # has a "nature" feature (3rd number)
rob   = [0, 2, 0, 1]      # has a "money/crime" feature (4th number)

print(contextualise([river, bank])[1])   # [1.5, 0.5, 0.5, 0.0]  picks up "nature"
print(contextualise([rob, bank])[1])     # [0.5, 1.5, 0.0, 0.5]  picks up "money/crime"
```

**What happened:** "bank" started identical both times, but after attention it took on features from its neighbour, so it ended up with two different meanings.

## Watch Out
- Transformers learn patterns in text extremely well, but whether that equals human "understanding" is debated.
- Attention alone isn't the whole story: scale, data quality and extra training matter a lot too.

Note: Q3 is about the historical impact of Transformers; this question is about why the design fits language.

## Remember
- Attention links any two words directly.
- Each word's meaning is rebuilt from its context.
- Many heads + many layers capture many kinds of structure.

## Related Questions
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [What is multi-head attention?](../Week-04-Transformer-Architecture/08-What-is-multi-head-attention.md)
- [Why did Transformers become important?](../Week-04-Transformer-Architecture/03-Why-did-Transformers-become-important.md)
- [What problems do traditional RNNs have?](../Week-04-Transformer-Architecture/01-What-problems-do-traditional-RNNs-have.md)
