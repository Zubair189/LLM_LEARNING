# What is a probability distribution?

> Week 03 – Mathematics and Neural Network Foundations · Question 7

## In One Line
A probability distribution is a **full list of every possible outcome and its chance**, where all the chances add up to 1.

## Simple Explanation
A single probability answers "how likely is **this one** outcome?". A distribution answers "how likely is **each** outcome?".

For a fair die, the distribution is:
- 1 → 1/6, 2 → 1/6, … 6 → 1/6. Together they add up to 1.

A **loaded** (unfair) die might be `[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]`: six is much more likely, but the total is still 1.

There are two main kinds:
1. **Discrete:** outcomes you can count, like die faces or the next word.
2. **Continuous:** outcomes on a smooth scale, like people's height. The famous "bell curve" (normal distribution) is one of these.

**Entropy** is a word for how **uncertain** a distribution is. If one outcome has almost all the probability, entropy is low (confident). If everything is equally likely, entropy is high (unsure).

**Link to LLMs:** at every step, an LLM outputs a distribution over its whole vocabulary (often 100,000+ word pieces). Then it **picks** a word from it:
- **Greedy:** always take the most likely word.
- **Sampling:** pick randomly, following the chances. Settings like temperature, top-k and top-p change how this pick is made.

## Real-Life Example
Think of a spinner wheel at a fair. Bigger slices are more likely to win. The sizes of all the slices make up the whole wheel (100%). An LLM's next-word choice is like a spinner whose slice sizes change with every sentence.

## Code Example
```python
import random

words = ["cat", "dog", "the", "sat"]
probs = [0.5, 0.3, 0.15, 0.05]        # one chance per word
print(sum(probs))                     # 1.0  (a valid distribution)

# Greedy: pick the most likely word
print(words[probs.index(max(probs))]) # cat

# Sampling: pick randomly, following the chances
random.seed(0)                        # fixed seed so the result repeats
print(random.choices(words, weights=probs, k=5))
# ['the', 'dog', 'cat', 'cat', 'dog']
```

**What happened:** the list of chances is the distribution. Greedy always picks "cat", while sampling mostly picks likely words but sometimes others.

## Watch Out
- **Logits** (the raw scores a model produces) are **not** a distribution. They can be negative and don't add up to 1. Softmax turns them into one.
- If you remove some options (as top-k does), you must rescale the rest so they add up to 1 again.

## Remember
- A distribution gives a chance to **every** outcome; they sum to 1.
- LLMs output a distribution over all possible next tokens.
- Decoding = choosing a token from that distribution.

## Related Questions
- [What is probability?](../Week-03-Mathematics-and-Neural-Network-Foundations/06-What-is-probability.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
