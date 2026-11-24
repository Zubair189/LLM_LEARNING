# What problems do traditional RNNs have?

> Week 04 – Transformer Architecture · Question 1

## In One Line
RNNs read text **one word at a time** and squeeze everything into a small memory, so they **forget early words, train slowly, and struggle with long text**.

## Simple Explanation
An **RNN** (Recurrent Neural Network) was the main way to process text before Transformers. It reads a sentence word by word. After each word, it updates a small **hidden state**, a fixed-size list of numbers that acts as its "memory so far".

```
word1 → [RNN] → memory1 → [RNN] → memory2 → [RNN] → memory3 → ...
```

This design causes three big problems:

1. **It forgets early words (vanishing gradients).** During training, the learning signal has to travel backwards through every step. At each step it gets multiplied by a number. If that number is below 1, the signal shrinks to almost nothing; if above 1, it explodes. So the model can't learn links between words that are far apart.
2. **It is slow to train.** Word 5 can't be processed until word 4 is done. GPUs are great at doing many things at once, but here they mostly wait.
3. **Its memory is too small (a bottleneck).** A whole long sentence or document has to fit into one fixed-size memory. Details get lost.

**Partial fixes:** LSTMs and GRUs (RNNs with "gates" that control what to keep or forget) helped with forgetting, but were still slow and step-by-step.

**Link to LLMs:** Transformers (2017) solved all three by using **attention**: every word can look directly at every other word, and all words are processed at the same time during training.

## Real-Life Example
The game of "telephone". A message is whispered down a long line of people. By the end, details from the start are lost, and you can't speed it up because each person must wait for the one before.

## Code Example
What happens to a learning signal multiplied by the same number again and again (once per word):

```python
for w in [0.5, 0.9, 1.1]:                       # the number it is multiplied by
    print(w, [round(w ** T, 4) for T in (10, 50)])   # after 10 and 50 steps
# 0.5 [0.001, 0.0]          shrinks to nothing (vanishing)
# 0.9 [0.3487, 0.0052]      shrinks more slowly, still vanishes
# 1.1 [2.5937, 117.3909]    grows out of control (exploding)
```

**What happened:** after 50 steps (about 50 words), the signal either disappeared or blew up. That is why RNNs struggle to connect words far apart.

## Watch Out
- RNNs *can* read long text; the problem is **learning** connections across long distances.
- LSTMs improved things a lot but did not fix the slow, one-at-a-time processing.

## Remember
- RNNs read one word at a time with a small memory.
- Problems: forgetting, slow training, memory bottleneck.
- Transformers fix these with attention and parallel processing.

## Related Questions
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
- [Why did Transformers become important?](../Week-04-Transformer-Architecture/03-Why-did-Transformers-become-important.md)
- [What is attention?](../Week-04-Transformer-Architecture/04-What-is-attention.md)
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
