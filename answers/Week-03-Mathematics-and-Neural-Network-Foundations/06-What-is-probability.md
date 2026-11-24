# What is probability?

> Week 03 – Mathematics and Neural Network Foundations · Question 6

## In One Line
Probability is a number **between 0 and 1** that says how likely something is: 0 = impossible, 1 = certain.

## Simple Explanation
Probability measures **how likely** an event is.

- `0` means it will never happen.
- `0.5` means a 50/50 chance, like a coin flip landing heads.
- `1` means it will definitely happen.

**Basic rules:**
1. A probability is always between 0 and 1.
2. The probabilities of **all possible outcomes add up to 1**. A die has 6 sides, each with probability 1/6, and 6 × 1/6 = 1.
3. The chance something does **not** happen is `1 − P`. If rain is 0.7, no rain is 0.3.

**Conditional probability** means "the chance of A, **given** that B already happened". We write it `P(A | B)`. For example, the chance of the word "Paris" given the text "The capital of France is".

**Chaining probabilities:** the chance of a whole sentence is the chance of the first word × the chance of the second word given the first × and so on. Because multiplying many small numbers gives a tiny number, computers often add up **logarithms** (log-probabilities) instead.

**Link to LLMs:** an LLM is a probability machine. At each step it gives a probability to every possible next word, then picks one. P("Paris" | "The capital of France is") might be 0.92.

## Real-Life Example
A weather forecast says "70% chance of rain". That means on days like this, it rained about 7 times out of 10. An LLM saying "92% chance the next word is Paris" is the same kind of statement, but about text.

## Code Example
```python
import math

p_the = 0.05              # P("the")
p_cat = 0.01              # P("cat" | "the")
p_sat = 0.2               # P("sat" | "the cat")

# Chance of the whole sentence "the cat sat"
p_sentence = p_the * p_cat * p_sat
print(round(p_sentence, 6))            # 0.0001

# Same thing using logs (adding instead of multiplying)
log_p = math.log(p_the) + math.log(p_cat) + math.log(p_sat)
print(round(log_p, 2))                 # -9.21
```

**What happened:** we multiplied next-word probabilities to score a sentence. The answer is tiny, so we also showed the log version, which adds numbers and avoids getting too close to zero.

## Watch Out
- A high probability does **not** mean something is true. An LLM can be very "confident" about a wrong answer (a hallucination).
- `P(A | B)` is not the same as `P(B | A)`. P(wet ground | rain) is high, but P(rain | wet ground) may not be, because sprinklers exist.

## Remember
- Probability: 0 (impossible) to 1 (certain); all outcomes sum to 1.
- LLMs predict P(next word | words so far).
- Use logs when multiplying many small probabilities.

## Related Questions
- [What is a probability distribution?](../Week-03-Mathematics-and-Neural-Network-Foundations/07-What-is-a-probability-distribution.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
